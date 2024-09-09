use gategen::boolvar::*;
use gategen::dynintvar::*;
use gategen::intvar::*;
use infmachine_gen::*;

use std::fmt::Debug;

// Utilities for machine.
// General utitities for machine creation. Utilities designed to be generic and usable
// on machine with any value of parameters - cell_len_bits and data_part_len, proc_num...
// possible smallest cell_len_bits is 0 (cell_len=1), possible smallest data_part_len is 1.
//
// Basic utilities:
// * load and determine max position at proc_id.
// * move back to start position in any data.
// * move data to another data (example: proc_id to mem_address or temp_buffer).
// * process data with any function.
// * any data operation includes stride (number of movements to next data part).
// * process integer in memory in ENDFORM form: [NP0, END0, NP1, END1, NP2, END2, ..].
//
// Number in ENDFORM: [NP0, END0, NP1, END1, NP2, END2, ..].
// NPx - part of number from lowest to highest.
// ENDx - mark last number part if value is not zero.
// Form: configurable circuit stage with specified part of stage indicator.
//       [STAGE_INDICATOR, STAGE_STATE, UNUSED]
// END_DP_FORM: [NP0_0, NP0_1, .., NP0_T, END0, NP1_0, NP1_1, .., NP1_T, END1, .....].
// T - (data_part_len + cell_len - 1) / cell_len.  Number of required cells to store data part.
//
// Temp buffer organization:
// data_part_len > 1: [XEND0,X0_0,X1_0,X2_0,....,XEND1,X0_1,X1_1,X2_1,.....,
//                     XEND2,X0_2,X1_2,X2_2......]
// XENDx - hold end position marker (end_pos):
//        bit 0 - memory end position marker, bit 1 - proc id end position marker.
//        If bit of given end marker is 1 - then no more data of given type of data starting
//        from this place.
// Xx_y - y'th data part of data x. Number of data parts will be determined by designer.
//
// data_part_len = 1: [MEND0,PEND0,X0_0,X1_0,X2_0,....,MEND1,PEND1,X0_1,X1_1,X2_1,.....,
//                     MEND2,PEND2,X0_2,X1_2,X2_2......]
// MENDx - hold memory end position marker. If bit is 1 then no more data in data starting
//        from this place.
// PENDx - hold proc_id end position marker. If bit is 1 then no more data in data starting
//        from this place.
// Xx_y - y'th data part of data x. Number of data parts will be determined by designer.
//
// temp_buffer_step - Number of datas including end position markers.
//
// Limiter: end position marker datas.
// Placed in first temp buffer data_part_len bit words.
// Temp buffer chunk part: [WORD0, WORD1,...]
// WORD0: 0 bit - memory address end pos, 1 bit - proc id end pos, 2 bit - other end pos, ....

const fn calc_log_bits(n: usize) -> usize {
    let nbits = usize::BITS - n.leading_zeros();
    if (1 << (nbits - 1)) == n {
        (nbits - 1) as usize
    } else {
        nbits as usize
    }
}

const fn calc_log_bits_u64(n: u64) -> usize {
    let nbits = u64::BITS - n.leading_zeros();
    if (1 << (nbits - 1)) == n {
        (nbits - 1) as usize
    } else {
        nbits as usize
    }
}

pub fn extend_output_state(state_start: usize, extra_bits: usize, input: &mut InfParInputSys) {
    assert!(state_start <= input.state.bitnum());
    if state_start + extra_bits > input.state.bitnum() {
        input.state = input.state.clone().concat(UDynVarSys::var(
            state_start + extra_bits - input.state.bitnum(),
        ));
    } else {
        input.state = input.state.clone();
    }
}

// return 1 bit state to handle unused bits
pub fn unused_inputs(mobj: &InfParMachineObjectSys, input_state: BoolVarSys) -> BoolVarSys {
    &input_state
        | mobj
            .in_memval
            .iter()
            .fold(BoolVarSys::from(false), |a, x| a.clone() | x.clone())
        | mobj
            .in_dpval
            .iter()
            .fold(BoolVarSys::from(false), |a, x| a.clone() | x.clone())
        | mobj.in_dp_move_done.clone()
}

// join_stage zeroes stage stage at end and allow self looping.
pub fn join_stage(
    next_state: UDynVarSys,
    mut output: InfParOutputSys,
    end: BoolVarSys,
) -> InfParOutputSys {
    let state_start = next_state.bitnum();
    let old_state = output.state.clone().subvalue(0, state_start);
    if output.state.bitnum() != state_start {
        let state_stage = output
            .state
            .clone()
            .subvalue(state_start, output.state.bitnum() - state_start);
        output.state = dynint_ite(
            end.clone(),
            next_state.concat(UDynVarSys::from_n(0u8, output.state.bitnum() - state_start)),
            old_state.concat(state_stage),
        );
        output
    } else {
        output.state = dynint_ite(end.clone(), next_state, old_state);
        output
    }
}

pub fn finish_stage_with_table(
    output_state: UDynVarSys,
    next_state: UDynVarSys,
    input: &InfParInputSys,
    mut output_stages: Vec<InfParOutputSys>,
    stage: UDynVarSys,
    end: BoolVarSys,
) -> (InfParOutputSys, BoolVarSys) {
    let state_start = output_state.bitnum();
    InfParOutputSys::fix_state_len(&mut output_stages);
    let output_stages = output_stages
        .into_iter()
        .map(|v| {
            let state_int = v.to_dynintvar();
            state_int.subvalue(state_start, state_int.bitnum() - state_start)
        })
        .collect::<Vec<_>>();
    let last = UDynVarSys::from_n(0u8, output_stages[0].bitnum());
    // Use output state outside joining outputs to reduce gates. It is possible because
    // first outputs are state outputs.
    let final_state = output_state.concat(dynint_table_partial(stage, output_stages, last));
    let output = InfParOutputSys::new_from_dynintvar(input.config(), final_state);
    (join_stage(next_state, output, end.clone()), end)
}

// function form: f(output_state, UDynVarSys, state_start: usize, in_output: &InfParOutputSys)
//                -> (UDynVarSys, InfParOutputSys)
// arguments:
// output_state - output_state of state_start length that choose this stage
// next_state - next state of state_start length that choose if end
// input - InfParInput with input state and circuit inputs.
// return:
// (input_full_state, output, end condition):
//   input_full_state - full input state with input state for this stage.
//   output - output InfParOutputSys
//   end condition - condition if stage ends
//
// Stage behavior:
// Initial state for stage is 0.
// At last stage step all extra_bits including unused SHOULD BE be cleared.

pub fn move_data_pos_stage(
    output_state: UDynVarSys,
    next_state: UDynVarSys,
    input: &mut InfParInputSys,
    data_kind: u8,
    dpmove: u8,
    step_num: u64,
) -> (InfParOutputSys, BoolVarSys) {
    assert_eq!(output_state.bitnum(), next_state.bitnum());
    let state_start = output_state.bitnum();
    let step_num_bits = calc_log_bits_u64(step_num);
    extend_output_state(state_start, step_num_bits, input);
    let input: &_ = input;
    let mut output = InfParOutputSys::new(input.config());
    let end = if step_num_bits != 0 {
        let in_step = input.state.subvalue(state_start, step_num_bits);
        let end = (&in_step).equal(step_num - 1);
        output.state = output_state.clone().concat(&in_step + 1u8);
        end
    } else {
        output.state = output_state.clone();
        true.into()
    };
    output.dkind = U2VarSys::from(data_kind);
    output.dpmove = U2VarSys::from(dpmove);
    (join_stage(next_state, output, end.clone()), end)
}

// step_num_m1 - step_num - 1
pub fn move_data_pos_expr_stage(
    output_state: UDynVarSys,
    next_state: UDynVarSys,
    input: &mut InfParInputSys,
    data_kind: u8,
    dpmove: u8,
    step_num_m1: UDynVarSys,
) -> (InfParOutputSys, BoolVarSys) {
    assert_eq!(output_state.bitnum(), next_state.bitnum());
    let state_start = output_state.bitnum();
    let step_num_bits = step_num_m1.bitnum();
    extend_output_state(state_start, step_num_bits, input);
    let input: &_ = input;
    let mut output = InfParOutputSys::new(input.config());
    let in_step = input.state.subvalue(state_start, step_num_bits);
    let end = (&in_step).equal(step_num_m1);
    output.state = output_state.clone().concat(&in_step + 1u8);
    output.dkind = U2VarSys::from(data_kind);
    output.dpmove = U2VarSys::from(dpmove);
    (join_stage(next_state, output, end.clone()), end)
}

pub fn data_pos_to_start_stage(
    output_state: UDynVarSys,
    next_state: UDynVarSys,
    input: &mut InfParInputSys,
    data_kind: u8,
) -> (InfParOutputSys, BoolVarSys) {
    assert_eq!(output_state.bitnum(), next_state.bitnum());
    let state_start = output_state.bitnum();
    extend_output_state(state_start, 1, input);
    let input: &_ = input;
    let end = input.state.bit(state_start) & !&input.dp_move_done;
    let mut output = InfParOutputSys::new(input.config());
    output.state = output_state.concat(UDynVarSys::from_n(1u8, 1));
    output.dkind = U2VarSys::from(data_kind);
    output.dpmove = U2VarSys::from(DPMOVE_BACKWARD);
    (join_stage(next_state, output, end.clone()), end)
}

// sequential increase memory address stage -
// sequential - only if all processors have this same memory address.
pub fn seq_increase_mem_address_stage(
    output_state: UDynVarSys,
    next_state: UDynVarSys,
    input: &mut InfParInputSys,
) -> (InfParOutputSys, BoolVarSys) {
    assert_eq!(output_state.bitnum(), next_state.bitnum());
    let state_start = output_state.bitnum();
    // 1. load data part from mem_address.
    // 2. Increase data part value and store to mem_address.
    // 3. If carry after increasing value then:
    // 3.1. Move forward (increase mem_address_pos) in mem_address and go to 1.
    // 4. Otherwise Move mem_address_pos back.
    extend_output_state(state_start, 2, input);
    let stage = U2VarSys::try_from(input.state.clone().subvalue(state_start, 2)).unwrap();
    let output_base = InfParOutputSys::new(input.config());
    // Stage 0b00. 1. load data part from mem_address.
    let mut output_0 = output_base.clone();
    output_0.state = output_state.clone().concat(U2VarSys::from(1u8).into());
    output_0.dpr = true.into();
    // Stage 0b01. 2. Increase data part value and store to mem_address.
    // Stage 0b01. 3. If carry after increasing value then:
    // Stage 0b01. 3.1. Move forward (increase mem_address_pos) in mem_address and go to 1.
    let mut output_1 = output_base.clone();
    let (new_value, carry) = input.dpval.addc_with_carry(
        &UDynVarSys::from_n(1u8, input.dpval.bitnum()),
        &false.into(),
    );
    output_1.state = output_state
        .clone()
        .concat(int_ite(carry.clone(), U2VarSys::from(0u8), U2VarSys::from(2u8)).into());
    output_1.dpmove = int_ite(
        carry,
        U2VarSys::from(DPMOVE_FORWARD),
        U2VarSys::from(DPMOVE_NOTHING),
    );
    output_1.dpw = true.into(); // store value to data part
    output_1.dpval = new_value;
    // Stage 0b10. 4. Otherwise Move mem_address_pos back.
    let (output_2, end) = data_pos_to_start_stage(
        output_state.clone().concat(U2VarSys::from(2u8).into()),
        output_state.clone().concat(U2VarSys::from(0u8).into()),
        input,
        DKIND_MEM_ADDRESS,
    );
    let end = (&stage).equal(U2VarSys::from(2u8)) & end;
    let output_stages = vec![output_0, output_1, output_2];
    finish_stage_with_table(
        output_state,
        next_state,
        input,
        output_stages,
        stage.into(),
        end,
    )
}

// init_mem_address_end_pos - initialize memory address end position from memory.
// Information about MemAddressEndPos in memory:
// At memory address 0: sequences of values between 1..=MAX and one zero,
// MemAddressPosEndPos is sum of non-zero cells.

// init_proc_id_end_pos - initialize proc id end position from memory.
// Information about ProcIdEndPos in memory: This same as in MemAddressEndPos and start
// after MemAddressEndPos in memory.

// Join together init_mem_address_end_pos and init_proc_id_end_pos.
// First is mem_address_end_pos, second is proc_id_pos.
pub fn init_machine_end_pos_stage(
    output_state: UDynVarSys,
    next_state: UDynVarSys,
    input: &mut InfParInputSys,
    temp_buffer_step: u32,
) -> (InfParOutputSys, BoolVarSys) {
    assert_eq!(output_state.bitnum(), next_state.bitnum());
    assert_ne!(temp_buffer_step, 0);
    let config = input.config();
    let cell_len = 1 << config.cell_len_bits;
    let state_start = output_state.bitnum();
    type StageType = U4VarSys;
    extend_output_state(state_start, StageType::BITS + 1 + cell_len, input);
    let stage =
        StageType::try_from(input.state.clone().subvalue(state_start, StageType::BITS)).unwrap();
    let is_proc_id = input.state.bit(state_start + StageType::BITS);
    let value_count = input
        .state
        .clone()
        .subvalue(state_start + StageType::BITS + 1, cell_len);
    let output_base = InfParOutputSys::new(config);
    let create_out_state = |s: StageType, ip, v| {
        output_state
            .clone()
            .concat(s.into())
            .concat(UDynVarSys::filled(1, ip))
            .concat(v)
    };
    // Stages:
    // tidx - stage index for main routine
    let tidx = if config.data_part_len <= 1 {
        assert!(temp_buffer_step >= 2);
        1u8
    } else {
        0u8
    };
    // make temp buffer position to 1.
    let mut output_tshift = output_base.clone();
    if config.data_part_len == 1 {
        output_tshift.state = create_out_state(
            StageType::from(1u8),
            is_proc_id.clone(),
            UDynVarSys::from_n(0u8, cell_len),
        );
        output_tshift.dpmove = int_ite(
            is_proc_id.clone(),
            U2VarSys::from(DPMOVE_FORWARD),
            U2VarSys::from(DPMOVE_NOTHING),
        );
        output_tshift.dkind = DKIND_TEMP_BUFFER.into();
    }
    // 0: 1. Load cell from memory.
    let mut output_0 = output_base.clone();
    output_0.state = create_out_state(
        StageType::from(tidx + 1u8),
        is_proc_id.clone(),
        UDynVarSys::from_n(0u8, cell_len),
    );
    output_0.memr = true.into();
    // 1: 2. If cell==0 then end go to 5.
    let mut output_1 = output_base.clone();
    output_1.state = create_out_state(
        int_ite(
            (&input.memval).equal(0u8),
            // end of algorithm
            StageType::from(tidx + 5u8),
            // start move temp buffer position
            StageType::from(tidx + 2u8),
        ),
        is_proc_id.clone(),
        input.memval.clone(),
    );
    // 3. If cell!=0 then:
    // 3.1. Decrease this value.
    let mut output_2 = output_base.clone();
    output_2.state = create_out_state(
        StageType::from(tidx + 3u8),
        is_proc_id.clone(),
        &value_count - 1u8,
    );
    // 3.2. Add temp_buffer_step to temp_buffer_pos
    let next_stage_3 = int_ite(
        (&value_count).equal(0u8),
        // if end of value_count then increase mem address
        StageType::from(tidx + 4u8),
        // continue
        StageType::from(tidx + 2u8),
    );
    // 4. If cell==0 then:
    let (output_3, _) = move_data_pos_stage(
        create_out_state(
            StageType::from(tidx + 3u8),
            is_proc_id.clone(),
            value_count.clone(),
        ),
        create_out_state(next_stage_3, is_proc_id.clone(), value_count.clone()),
        input,
        DKIND_TEMP_BUFFER,
        DPMOVE_FORWARD,
        temp_buffer_step as u64,
    );
    // 4.1. increase memory_address, load cell from memory and go to 2.
    // 5. increase memory_address.
    let (mut output_4, end_4) = seq_increase_mem_address_stage(
        create_out_state(stage.clone(), is_proc_id.clone(), value_count.clone()),
        create_out_state(
            int_ite(
                (&stage).equal(tidx + 5u8),
                StageType::from(tidx + 6u8),
                StageType::from(tidx + 1u8),
            ),
            is_proc_id.clone(),
            value_count.clone(),
        ),
        input,
    );
    // at end read memory
    output_4.memr = end_4;
    // 6. Set 1 to current temp buffer part.
    let (output_6, output_6_1, tidx) = if config.data_part_len > 1 {
        // if data_part_len > 1: read temp buffer part
        let mut output_6 = output_base.clone();
        output_6.state = create_out_state(
            StageType::from(tidx + 7u8),
            is_proc_id.clone(),
            UDynVarSys::from_n(0u8, cell_len),
        );
        output_6.dkind = DKIND_TEMP_BUFFER.into();
        output_6.dpr = true.into();
        // and or value with current data part and 2
        let mut output_6_1 = output_base.clone();
        output_6_1.state = create_out_state(
            StageType::from(tidx + 8u8),
            is_proc_id.clone(),
            UDynVarSys::from_n(0u8, cell_len),
        );
        output_6_1.dpval = dynint_ite(
            is_proc_id.clone(),
            UDynVarSys::from_n(2u8, config.data_part_len as usize),
            UDynVarSys::from_n(1u8, config.data_part_len as usize),
        ) | &input.dpval;
        output_6_1.dkind = DKIND_TEMP_BUFFER.into();
        output_6_1.dpw = true.into();
        (output_6, output_6_1, tidx + 1)
    } else {
        // if data_part_len == 1: Set 1 to current temp buffer part.
        let mut output_6 = output_base.clone();
        output_6.state = create_out_state(
            StageType::from(tidx + 7u8),
            is_proc_id.clone(),
            UDynVarSys::from_n(0u8, cell_len),
        );
        output_6.dpval = UDynVarSys::from_n(1u8, config.data_part_len as usize);
        output_6.dkind = DKIND_TEMP_BUFFER.into();
        output_6.dpw = true.into();
        (output_6.clone(), output_6, tidx)
    };
    // 7. Move temp buffer part pos to start.
    let (output_7, end_7) = data_pos_to_start_stage(
        create_out_state(
            StageType::from(tidx + 7u8),
            is_proc_id.clone(),
            UDynVarSys::from_n(0u8, cell_len),
        ),
        create_out_state(
            StageType::from(0u8),
            !&is_proc_id,
            UDynVarSys::from_n(0u8, cell_len),
        ),
        input,
        DKIND_TEMP_BUFFER,
    );
    let end = is_proc_id & end_7 & (&stage).equal(tidx + 7u8);
    // finishing
    let output_stages = if config.data_part_len > 1 {
        vec![
            output_0,
            output_1,
            output_2,
            output_3,
            output_4.clone(),
            output_4,
            output_6,
            output_6_1,
            output_7,
        ]
    } else {
        vec![
            output_tshift,
            output_0,
            output_1,
            output_2,
            output_3,
            output_4.clone(),
            output_4,
            output_6,
            output_7,
        ]
    };
    // extend to 16 elements
    finish_stage_with_table(
        output_state,
        next_state,
        input,
        output_stages,
        stage.into(),
        end,
    )
}

// function parameters in infinite data (memaddrss, tempbuffer, procids).

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum InfDataParam {
    MemAddress,
    ProcId,
    TempBuffer(usize),
    EndPos(usize),
}

// define default end position markers
pub const END_POS_MEM_ADDRESS: usize = 0;
pub const END_POS_PROC_ID: usize = 1;

// functions

pub trait Function1 {
    fn state_len(&self) -> usize;
    // return (output state, output)
    fn output(&self, input_state: UDynVarSys, i0: UDynVarSys) -> (UDynVarSys, UDynVarSys);
}

pub trait Function2 {
    fn state_len(&self) -> usize;
    // return (output state, output)
    fn output(
        &self,
        input_state: UDynVarSys,
        i0: UDynVarSys,
        i1: UDynVarSys,
    ) -> (UDynVarSys, UDynVarSys);
}

pub trait Function2_2 {
    fn state_len(&self) -> usize;
    // return (output state, output0, output1)
    fn output(
        &self,
        input_state: UDynVarSys,
        i0: UDynVarSys,
        i1: UDynVarSys,
    ) -> (UDynVarSys, UDynVarSys, UDynVarSys);
}

pub trait FunctionNN {
    fn state_len(&self) -> usize;
    fn input_num(&self) -> usize;
    fn output_num(&self) -> usize;
    // return (output state, outputs_vec)
    fn output(
        &self,
        input_state: UDynVarSys,
        inputs: &[UDynVarSys],
    ) -> (UDynVarSys, Vec<UDynVarSys>);
}

pub struct FuncNNAdapter1<F: Function1> {
    f: F,
}

impl<F: Function1> From<F> for FuncNNAdapter1<F> {
    fn from(f: F) -> Self {
        Self { f }
    }
}

pub struct FuncNNAdapter2<F: Function2> {
    f: F,
}

impl<F: Function2> From<F> for FuncNNAdapter2<F> {
    fn from(f: F) -> Self {
        Self { f }
    }
}

pub struct FuncNNAdapter2_2<F: Function2_2> {
    f: F,
}

impl<F: Function2_2> From<F> for FuncNNAdapter2_2<F> {
    fn from(f: F) -> Self {
        Self { f }
    }
}

impl<F: Function1> FunctionNN for FuncNNAdapter1<F> {
    fn state_len(&self) -> usize {
        self.f.state_len()
    }
    fn input_num(&self) -> usize {
        1
    }
    fn output_num(&self) -> usize {
        1
    }
    fn output(
        &self,
        input_state: UDynVarSys,
        inputs: &[UDynVarSys],
    ) -> (UDynVarSys, Vec<UDynVarSys>) {
        let (out_state, output) = self.f.output(input_state, inputs[0].clone());
        (out_state, vec![output])
    }
}

impl<F: Function2> FunctionNN for FuncNNAdapter2<F> {
    fn state_len(&self) -> usize {
        self.f.state_len()
    }
    fn input_num(&self) -> usize {
        2
    }
    fn output_num(&self) -> usize {
        1
    }
    fn output(
        &self,
        input_state: UDynVarSys,
        inputs: &[UDynVarSys],
    ) -> (UDynVarSys, Vec<UDynVarSys>) {
        let (out_state, output) = self
            .f
            .output(input_state, inputs[0].clone(), inputs[1].clone());
        (out_state, vec![output])
    }
}

impl<F: Function2_2> FunctionNN for FuncNNAdapter2_2<F> {
    fn state_len(&self) -> usize {
        self.f.state_len()
    }
    fn input_num(&self) -> usize {
        2
    }
    fn output_num(&self) -> usize {
        2
    }
    fn output(
        &self,
        input_state: UDynVarSys,
        inputs: &[UDynVarSys],
    ) -> (UDynVarSys, Vec<UDynVarSys>) {
        let (out_state, output, output2) =
            self.f
                .output(input_state, inputs[0].clone(), inputs[1].clone());
        (out_state, vec![output, output2])
    }
}

pub struct Copy1Func {}

impl Copy1Func {
    pub fn new() -> Self {
        Self {}
    }
}

impl Function1 for Copy1Func {
    fn state_len(&self) -> usize {
        0
    }
    fn output(&self, _: UDynVarSys, i0: UDynVarSys) -> (UDynVarSys, UDynVarSys) {
        (UDynVarSys::var(0), i0)
    }
}

pub struct Add1Func {
    inout_len: usize,
    value: UDynVarSys,
}

impl Add1Func {
    pub fn new(inout_len: usize, value: UDynVarSys) -> Self {
        Self { inout_len, value }
    }
    pub fn new_from_u64(inout_len: usize, value: u64) -> Self {
        Self {
            inout_len,
            value: if value != 0 {
                UDynVarSys::from_n(value, (u64::BITS - value.leading_zeros()) as usize)
            } else {
                UDynVarSys::from_n(value, 1)
            },
        }
    }
}

impl Function1 for Add1Func {
    fn state_len(&self) -> usize {
        calc_log_bits(((self.value.bitnum() + self.inout_len - 1) / self.inout_len) + 1) + 1
    }
    fn output(&self, input_state: UDynVarSys, i0: UDynVarSys) -> (UDynVarSys, UDynVarSys) {
        let max_state_count = (self.value.bitnum() + self.inout_len - 1) / self.inout_len;
        let state_len = self.state_len();
        // get current part of value to add to input.
        let index = input_state.clone().subvalue(0, state_len - 1);
        let old_carry = input_state.bit(state_len - 1);
        let adder = dynint_table_partial(
            index.clone(),
            (0..max_state_count).map(|i| {
                UDynVarSys::try_from_n(
                    self.value.subvalue(
                        i * self.inout_len,
                        std::cmp::min((i + 1) * self.inout_len, self.value.bitnum())
                            - i * self.inout_len,
                    ),
                    self.inout_len,
                )
                .unwrap()
            }),
            UDynVarSys::from_n(0u8, self.inout_len),
        );
        let (result, carry) = i0.addc_with_carry(&adder, &old_carry);
        let next_state = dynint_ite(
            (&index).equal(max_state_count),
            UDynVarSys::from_n(max_state_count, state_len - 1),
            &index + 1u8,
        )
        .concat(UDynVarSys::filled(1, carry));
        (next_state, result)
    }
}

pub struct XorNNFuncSample {
    inout_len: usize,
    input_num: usize,
    output_num: usize,
}

impl XorNNFuncSample {
    pub fn new(inout_len: usize, input_num: usize, output_num: usize) -> Self {
        Self {
            inout_len,
            input_num,
            output_num,
        }
    }
}

impl FunctionNN for XorNNFuncSample {
    fn state_len(&self) -> usize {
        0
    }
    fn input_num(&self) -> usize {
        self.input_num
    }
    fn output_num(&self) -> usize {
        self.output_num
    }
    fn output(&self, _: UDynVarSys, inputs: &[UDynVarSys]) -> (UDynVarSys, Vec<UDynVarSys>) {
        let mut outputs = (0..self.output_num)
            .map(|_| UDynVarSys::from_n(0u8, self.inout_len))
            .collect::<Vec<_>>();
        for i in 0..self.input_num {
            outputs[i % self.output_num] ^=
                UDynVarSys::try_from_n(inputs[i].clone(), self.inout_len).unwrap();
        }
        (UDynVarSys::var(0), outputs)
    }
}

// parallel routines

// temp_buffer_step - number of different datas in temp_buffer.
//                    number of step between next data part of same type.
// temp_buffer_step_pos - position of data in step: from 0 to temp_buffer_step - 1 inclusively.

pub fn par_copy_proc_id_to_temp_buffer_stage(
    output_state: UDynVarSys,
    next_state: UDynVarSys,
    input: &mut InfParInputSys,
    temp_buffer_step: u32,
    temp_buffer_step_pos: u32,
) -> (InfParOutputSys, BoolVarSys) {
    // par_process_proc_id_to_temp_buffer_stage(
    //     output_state,
    //     next_state,
    //     input,
    //     temp_buffer_step,
    //     temp_buffer_step_pos,
    //     Copy1Func::new(),
    // )
    par_process_infinite_data_stage(
        output_state,
        next_state,
        input,
        temp_buffer_step,
        &[(InfDataParam::ProcId, END_POS_PROC_ID)],
        &[(
            InfDataParam::TempBuffer(temp_buffer_step_pos as usize),
            END_POS_PROC_ID,
        )],
        FuncNNAdapter1::from(Copy1Func::new()),
    )
}

// par_copy_proc_id_to_mem_address_stage - copy proc_id to mem_address.
// Include mem_address_pos_end and include proc_id_pos_end.
// If mem_address_pos_end is greater then fill rest beyond proc_id_end_pos by zeroes.
pub fn par_copy_proc_id_to_mem_address_stage(
    output_state: UDynVarSys,
    next_state: UDynVarSys,
    input: &mut InfParInputSys,
    temp_buffer_step: u32,
) -> (InfParOutputSys, BoolVarSys) {
    par_process_infinite_data_stage(
        output_state,
        next_state,
        input,
        temp_buffer_step,
        &[(InfDataParam::ProcId, END_POS_PROC_ID)],
        &[(InfDataParam::MemAddress, END_POS_MEM_ADDRESS)],
        FuncNNAdapter1::from(Copy1Func::new()),
    )
}

// par_copy_temp_buffer_to_mem_address_stage - copy temp_buffer to mem_address
// Include mem_address_pos_end.
pub fn par_copy_temp_buffer_to_mem_address_stage(
    output_state: UDynVarSys,
    next_state: UDynVarSys,
    input: &mut InfParInputSys,
    temp_buffer_step: u32,
    temp_buffer_step_pos: u32,
) -> (InfParOutputSys, BoolVarSys) {
    par_process_infinite_data_stage(
        output_state,
        next_state,
        input,
        temp_buffer_step,
        &[(
            InfDataParam::TempBuffer(temp_buffer_step_pos as usize),
            END_POS_MEM_ADDRESS,
        )],
        &[(InfDataParam::MemAddress, END_POS_MEM_ADDRESS)],
        FuncNNAdapter1::from(Copy1Func::new()),
    )
}

// par_copy_mem_address_to_temp_buffer_stage - copy mem_address to temp_buffer
// Include mem_address_pos_end.
pub fn par_copy_mem_address_to_temp_buffer_stage(
    output_state: UDynVarSys,
    next_state: UDynVarSys,
    input: &mut InfParInputSys,
    temp_buffer_step: u32,
    temp_buffer_step_pos: u32,
) -> (InfParOutputSys, BoolVarSys) {
    par_process_infinite_data_stage(
        output_state,
        next_state,
        input,
        temp_buffer_step,
        &[(InfDataParam::MemAddress, END_POS_MEM_ADDRESS)],
        &[(
            InfDataParam::TempBuffer(temp_buffer_step_pos as usize),
            END_POS_MEM_ADDRESS,
        )],
        FuncNNAdapter1::from(Copy1Func::new()),
    )
}

// par_copy_temp_buffer_to_temp_buffer_stage - copy temp_buffer to temp_buffer
// proc_id_end_pos - if value true then use proc_id_end_pos to determine length
// otherwise use mem_address_end_pos to determine length.
pub fn par_copy_temp_buffer_to_temp_buffer_stage(
    output_state: UDynVarSys,
    next_state: UDynVarSys,
    input: &mut InfParInputSys,
    temp_buffer_step: u32,
    tbs_src_pos: u32,
    tbs_dest_pos: u32,
    proc_id_end_pos: bool,
) -> (InfParOutputSys, BoolVarSys) {
    let end_pos = if proc_id_end_pos {
        END_POS_PROC_ID
    } else {
        END_POS_MEM_ADDRESS
    };
    par_process_infinite_data_stage(
        output_state,
        next_state,
        input,
        temp_buffer_step,
        &[(InfDataParam::TempBuffer(tbs_src_pos as usize), end_pos)],
        &[(InfDataParam::TempBuffer(tbs_dest_pos as usize), end_pos)],
        FuncNNAdapter1::from(Copy1Func::new()),
    )
}

// process routines

// par_process_proc_id_to_temp_buffer_stage - process proc_id to temp buffer in specified pos.
// temp_buffer_step_pos - position chunk (specify data position).
pub fn par_process_proc_id_to_temp_buffer_stage<F: Function1>(
    output_state: UDynVarSys,
    next_state: UDynVarSys,
    input: &mut InfParInputSys,
    temp_buffer_step: u32,
    temp_buffer_step_pos: u32,
    func: F,
) -> (InfParOutputSys, BoolVarSys) {
    par_process_infinite_data_stage(
        output_state,
        next_state,
        input,
        temp_buffer_step,
        &[(InfDataParam::ProcId, END_POS_PROC_ID)],
        &[(
            InfDataParam::TempBuffer(temp_buffer_step_pos as usize),
            END_POS_PROC_ID,
        )],
        FuncNNAdapter1::from(func),
    )
}

// par_process_proc_id_to_mem_address_stage - process proc_id to mem_address.
// Include mem_address_pos_end and include proc_id_pos_end.
// If mem_address_pos_end is greater then fill rest beyond proc_id_end_pos by zeroes.
pub fn par_process_proc_id_to_mem_address_stage<F: Function1>(
    output_state: UDynVarSys,
    next_state: UDynVarSys,
    input: &mut InfParInputSys,
    temp_buffer_step: u32,
    func: F,
) -> (InfParOutputSys, BoolVarSys) {
    par_process_infinite_data_stage(
        output_state,
        next_state,
        input,
        temp_buffer_step,
        &[(InfDataParam::ProcId, END_POS_PROC_ID)],
        &[(InfDataParam::MemAddress, END_POS_MEM_ADDRESS)],
        FuncNNAdapter1::from(func),
    )
}

// par_process_temp_buffer_to_mem_address_stage - process temp_buffer to mem_address
// Include mem_address_pos_end.
pub fn par_process_temp_buffer_to_mem_address_stage<F: Function1>(
    output_state: UDynVarSys,
    next_state: UDynVarSys,
    input: &mut InfParInputSys,
    temp_buffer_step: u32,
    temp_buffer_step_pos: u32,
    func: F,
) -> (InfParOutputSys, BoolVarSys) {
    par_process_infinite_data_stage(
        output_state,
        next_state,
        input,
        temp_buffer_step,
        &[(
            InfDataParam::TempBuffer(temp_buffer_step_pos as usize),
            END_POS_MEM_ADDRESS,
        )],
        &[(InfDataParam::MemAddress, END_POS_MEM_ADDRESS)],
        FuncNNAdapter1::from(func),
    )
}

// par_process_mem_address_to_temp_buffer_stage - copy mem_address to temp_buffer
// Include mem_address_pos_end.
pub fn par_process_mem_address_to_temp_buffer_stage<F: Function1>(
    output_state: UDynVarSys,
    next_state: UDynVarSys,
    input: &mut InfParInputSys,
    temp_buffer_step: u32,
    temp_buffer_step_pos: u32,
    func: F,
) -> (InfParOutputSys, BoolVarSys) {
    par_process_infinite_data_stage(
        output_state,
        next_state,
        input,
        temp_buffer_step,
        &[(InfDataParam::MemAddress, END_POS_MEM_ADDRESS)],
        &[(
            InfDataParam::TempBuffer(temp_buffer_step_pos as usize),
            END_POS_MEM_ADDRESS,
        )],
        FuncNNAdapter1::from(func),
    )
}

// par_process_temp_buffer_to_temp_buffer_stage - process temp_buffer to temp_buffer
// proc_id_end_pos - if value true then use proc_id_end_pos to determine length
// otherwise use mem_address_end_pos to determine length.
pub fn par_process_temp_buffer_to_temp_buffer_stage<F: Function1>(
    output_state: UDynVarSys,
    next_state: UDynVarSys,
    input: &mut InfParInputSys,
    temp_buffer_step: u32,
    tbs_src_pos: u32,
    tbs_dest_pos: u32,
    src_proc_id_end_pos: bool,
    dest_proc_id_end_pos: bool,
    func: F,
) -> (InfParOutputSys, BoolVarSys) {
    par_process_infinite_data_stage(
        output_state,
        next_state,
        input,
        temp_buffer_step,
        &[(
            InfDataParam::TempBuffer(tbs_src_pos as usize),
            if src_proc_id_end_pos {
                END_POS_PROC_ID
            } else {
                END_POS_MEM_ADDRESS
            },
        )],
        &[(
            InfDataParam::TempBuffer(tbs_dest_pos as usize),
            if dest_proc_id_end_pos {
                END_POS_PROC_ID
            } else {
                END_POS_MEM_ADDRESS
            },
        )],
        FuncNNAdapter1::from(func),
    )
}

pub fn par_process_mem_address_stage<F: Function1>(
    output_state: UDynVarSys,
    next_state: UDynVarSys,
    input: &mut InfParInputSys,
    temp_buffer_step: u32,
    func: F,
) -> (InfParOutputSys, BoolVarSys) {
    par_process_infinite_data_stage(
        output_state,
        next_state,
        input,
        temp_buffer_step,
        &[(InfDataParam::MemAddress, END_POS_MEM_ADDRESS)],
        &[(InfDataParam::MemAddress, END_POS_MEM_ADDRESS)],
        FuncNNAdapter1::from(func),
    )
}

// OP(proc_id, temp_buffer) -> temp_buffer
pub fn par_process_proc_id_temp_buffer_to_temp_buffer_stage<F: Function2>(
    output_state: UDynVarSys,
    next_state: UDynVarSys,
    input: &mut InfParInputSys,
    temp_buffer_step: u32,
    src_tbs_pos: u32,
    dest_tbs_pos: u32,
    src_proc_id_end_pos: bool,
    dest_proc_id_end_pos: bool,
    func: F,
) -> (InfParOutputSys, BoolVarSys) {
    par_process_infinite_data_stage(
        output_state,
        next_state,
        input,
        temp_buffer_step,
        &[
            (InfDataParam::ProcId, END_POS_PROC_ID),
            (
                InfDataParam::TempBuffer(src_tbs_pos as usize),
                if src_proc_id_end_pos {
                    END_POS_PROC_ID
                } else {
                    END_POS_MEM_ADDRESS
                },
            ),
        ],
        &[(
            InfDataParam::TempBuffer(dest_tbs_pos as usize),
            if dest_proc_id_end_pos {
                END_POS_PROC_ID
            } else {
                END_POS_MEM_ADDRESS
            },
        )],
        FuncNNAdapter2::from(func),
    )
}

// OP(proc_id, mem_address) -> mem_address
pub fn par_process_proc_id_mem_address_to_mem_address_stage<F: Function2>(
    output_state: UDynVarSys,
    next_state: UDynVarSys,
    input: &mut InfParInputSys,
    temp_buffer_step: u32,
    func: F,
) -> (InfParOutputSys, BoolVarSys) {
    par_process_infinite_data_stage(
        output_state,
        next_state,
        input,
        temp_buffer_step,
        &[
            (InfDataParam::ProcId, END_POS_PROC_ID),
            (InfDataParam::MemAddress, END_POS_MEM_ADDRESS),
        ],
        &[(InfDataParam::MemAddress, END_POS_MEM_ADDRESS)],
        FuncNNAdapter2::from(func),
    )
}

// OP(proc_id, mem_address) -> temp_buffer
pub fn par_process_proc_id_mem_address_to_temp_buffer_stage<F: Function2>(
    output_state: UDynVarSys,
    next_state: UDynVarSys,
    input: &mut InfParInputSys,
    temp_buffer_step: u32,
    dest_tbs_pos: u32,
    dest_proc_id_end_pos: bool,
    func: F,
) -> (InfParOutputSys, BoolVarSys) {
    par_process_infinite_data_stage(
        output_state,
        next_state,
        input,
        temp_buffer_step,
        &[
            (InfDataParam::ProcId, END_POS_PROC_ID),
            (InfDataParam::MemAddress, END_POS_MEM_ADDRESS),
        ],
        &[(
            InfDataParam::TempBuffer(dest_tbs_pos as usize),
            if dest_proc_id_end_pos {
                END_POS_PROC_ID
            } else {
                END_POS_MEM_ADDRESS
            },
        )],
        FuncNNAdapter2::from(func),
    )
}

// OP(proc_id, temp_buffer) -> mem_address
pub fn par_process_proc_id_temp_buffer_to_mem_address_stage<F: Function2>(
    output_state: UDynVarSys,
    next_state: UDynVarSys,
    input: &mut InfParInputSys,
    temp_buffer_step: u32,
    src_tbs_pos: u32,
    src_proc_id_end_pos: bool,
    func: F,
) -> (InfParOutputSys, BoolVarSys) {
    par_process_infinite_data_stage(
        output_state,
        next_state,
        input,
        temp_buffer_step,
        &[
            (InfDataParam::ProcId, END_POS_PROC_ID),
            (
                InfDataParam::TempBuffer(src_tbs_pos as usize),
                if src_proc_id_end_pos {
                    END_POS_PROC_ID
                } else {
                    END_POS_MEM_ADDRESS
                },
            ),
        ],
        &[(InfDataParam::MemAddress, END_POS_MEM_ADDRESS)],
        FuncNNAdapter2::from(func),
    )
}

// OP(mem_address, temp_buffer) -> temp_buffer
pub fn par_process_mem_address_temp_buffer_to_temp_buffer_stage<F: Function2>(
    output_state: UDynVarSys,
    next_state: UDynVarSys,
    input: &mut InfParInputSys,
    temp_buffer_step: u32,
    src_tbs_pos: u32,
    dest_tbs_pos: u32,
    src_proc_id_end_pos: bool,
    dest_proc_id_end_pos: bool,
    func: F,
) -> (InfParOutputSys, BoolVarSys) {
    par_process_infinite_data_stage(
        output_state,
        next_state,
        input,
        temp_buffer_step,
        &[
            (InfDataParam::MemAddress, END_POS_MEM_ADDRESS),
            (
                InfDataParam::TempBuffer(src_tbs_pos as usize),
                if src_proc_id_end_pos {
                    END_POS_PROC_ID
                } else {
                    END_POS_MEM_ADDRESS
                },
            ),
        ],
        &[(
            InfDataParam::TempBuffer(dest_tbs_pos as usize),
            if dest_proc_id_end_pos {
                END_POS_PROC_ID
            } else {
                END_POS_MEM_ADDRESS
            },
        )],
        FuncNNAdapter2::from(func),
    )
}

// OP(mem_address, temp_buffer) -> mem_address
pub fn par_process_mem_address_temp_buffer_to_mem_address_stage<F: Function2>(
    output_state: UDynVarSys,
    next_state: UDynVarSys,
    input: &mut InfParInputSys,
    temp_buffer_step: u32,
    src_tbs_pos: u32,
    src_proc_id_end_pos: bool,
    func: F,
) -> (InfParOutputSys, BoolVarSys) {
    par_process_infinite_data_stage(
        output_state,
        next_state,
        input,
        temp_buffer_step,
        &[
            (InfDataParam::MemAddress, END_POS_MEM_ADDRESS),
            (
                InfDataParam::TempBuffer(src_tbs_pos as usize),
                if src_proc_id_end_pos {
                    END_POS_PROC_ID
                } else {
                    END_POS_MEM_ADDRESS
                },
            ),
        ],
        &[(InfDataParam::MemAddress, END_POS_MEM_ADDRESS)],
        FuncNNAdapter2::from(func),
    )
}

// OP(temp_buffer, temp_buffer) -> temp_buffer
pub fn par_process_temp_buffer_temp_buffer_to_temp_buffer_stage<F: Function2>(
    output_state: UDynVarSys,
    next_state: UDynVarSys,
    input: &mut InfParInputSys,
    temp_buffer_step: u32,
    src_tbs_pos: u32,
    src2_tbs_pos: u32,
    dest_tbs_pos: u32,
    src_proc_id_end_pos: bool,
    src2_proc_id_end_pos: bool,
    dest_proc_id_end_pos: bool,
    func: F,
) -> (InfParOutputSys, BoolVarSys) {
    par_process_infinite_data_stage(
        output_state,
        next_state,
        input,
        temp_buffer_step,
        &[
            (
                InfDataParam::TempBuffer(src_tbs_pos as usize),
                if src_proc_id_end_pos {
                    END_POS_PROC_ID
                } else {
                    END_POS_MEM_ADDRESS
                },
            ),
            (
                InfDataParam::TempBuffer(src2_tbs_pos as usize),
                if src2_proc_id_end_pos {
                    END_POS_PROC_ID
                } else {
                    END_POS_MEM_ADDRESS
                },
            ),
        ],
        &[(
            InfDataParam::TempBuffer(dest_tbs_pos as usize),
            if dest_proc_id_end_pos {
                END_POS_PROC_ID
            } else {
                END_POS_MEM_ADDRESS
            },
        )],
        FuncNNAdapter2::from(func),
    )
}

// OP(temp_buffer, temp_buffer) -> mem_address
pub fn par_process_temp_buffer_2_to_mem_address_stage<F: Function2>(
    output_state: UDynVarSys,
    next_state: UDynVarSys,
    input: &mut InfParInputSys,
    temp_buffer_step: u32,
    src_tbs_pos: u32,
    src2_tbs_pos: u32,
    src_proc_id_end_pos: bool,
    src2_proc_id_end_pos: bool,
    func: F,
) -> (InfParOutputSys, BoolVarSys) {
    par_process_infinite_data_stage(
        output_state,
        next_state,
        input,
        temp_buffer_step,
        &[
            (
                InfDataParam::TempBuffer(src_tbs_pos as usize),
                if src_proc_id_end_pos {
                    END_POS_PROC_ID
                } else {
                    END_POS_MEM_ADDRESS
                },
            ),
            (
                InfDataParam::TempBuffer(src2_tbs_pos as usize),
                if src2_proc_id_end_pos {
                    END_POS_PROC_ID
                } else {
                    END_POS_MEM_ADDRESS
                },
            ),
        ],
        &[(InfDataParam::MemAddress, END_POS_MEM_ADDRESS)],
        FuncNNAdapter2::from(func),
    )
}

macro_rules! test_println {
    () => { println!(); };
    ($($arg:tt)*) => { eprintln!($($arg)*); };
}

// macro_rules! test_println {
//     () => {};
//     ($($arg:tt)*) => {};
// }

// TODO: Test par_process_infinite_data_stage by using detailed debug prints.
// main routine to process infinite data (mem_address, proc_id and temp_buffer).
pub fn par_process_infinite_data_stage<F: FunctionNN>(
    output_state: UDynVarSys,
    next_state: UDynVarSys,
    input: &mut InfParInputSys,
    temp_buffer_step: u32,
    src_params: &[(InfDataParam, usize)],
    dests: &[(InfDataParam, usize)],
    func: F,
) -> (InfParOutputSys, BoolVarSys) {
    let src_len = src_params.len();
    let dest_len = dests.len();
    assert_eq!(output_state.bitnum(), next_state.bitnum());
    assert_eq!(func.input_num(), src_len);
    assert_eq!(func.output_num(), dest_len);
    let config = input.config();
    let dp_len = config.data_part_len as usize;
    // src_params can be empty (no input for functions)
    assert!(!dests.is_empty());
    for (data_param, end_pos) in src_params.iter().chain(dests.iter()) {
        let good = match data_param {
            InfDataParam::TempBuffer(pos) => *pos < temp_buffer_step as usize,
            InfDataParam::EndPos(idx) => *idx < dp_len * (temp_buffer_step as usize),
            _ => true,
        };
        assert!(good && *end_pos < dp_len * (temp_buffer_step as usize));
    }
    // words where is end position markers
    let end_pos_words = {
        let mut end_pos_words = src_params
            .iter()
            .chain(dests.iter())
            .filter_map(|(dp, _)| {
                if let InfDataParam::EndPos(pos) = dp {
                    // divide by data_part_len to get word position
                    Some(pos / dp_len)
                } else {
                    None
                }
            })
            .chain(
                src_params
                    .iter()
                    .chain(dests.iter())
                    .map(|(_, end_pos)| end_pos / dp_len),
            )
            .collect::<Vec<_>>();
        end_pos_words.sort();
        end_pos_words.dedup();
        end_pos_words
    };
    for (data_param, _) in src_params.into_iter().chain(dests.into_iter()) {
        if let InfDataParam::TempBuffer(pos) = data_param {
            // temp buffer positions shouldn't cover words with end pos markers
            assert!(end_pos_words.binary_search(pos).is_err());
        }
    }
    {
        let mut dests = dests.to_vec();
        dests.sort();
        let old_dest_len = dest_len;
        dests.dedup();
        // check whether dests have only one per different InfDataParam.
        assert_eq!(old_dest_len, dest_len);
    }
    assert!(dests
        .into_iter()
        .all(|(param, _)| *param != InfDataParam::ProcId));

    test_println!("par_process_infinite_data_stage:");
    test_println!("  SrcParams: {:?}", src_params);
    test_println!("  Dests: {:?}", dests);
    // check usage of other sources
    let use_mem_address = src_params
        .into_iter()
        .chain(dests.into_iter())
        .any(|(param, _)| *param == InfDataParam::MemAddress);
    let use_write_mem_address = dests
        .into_iter()
        .any(|(param, _)| *param == InfDataParam::MemAddress);
    let use_proc_id = src_params
        .into_iter()
        .any(|(param, _)| *param == InfDataParam::ProcId);
    test_println!(
        "  UseMemAddress: {}, UseWriteMemAdress: {}, UseProcId: {}",
        use_mem_address,
        use_write_mem_address,
        use_proc_id
    );

    let mut total_stages = 0;
    // store all end pos limiters
    let total_state_bits = src_len + dest_len;
    // end_pos
    let mut last_pos = 0;
    for list in [src_params, dests] {
        test_println!("  EndPosList: {:?}", list);
        let mut first = true;
        for (_, end_pos) in list {
            let pos = *end_pos / dp_len;
            if last_pos != pos {
                total_stages += 1; // movement
                total_stages += 2; // read stage and store stage
                test_println!("    EndPos: Move to last position: {} {}", last_pos, pos);
            } else if first {
                total_stages += 2; // read stage and store stage
            }
            first = false;
            last_pos = pos;
        }
        test_println!(
            "    EndPos: TotalStages: {}, TotalStateBits: {}, LastPos: {}",
            total_stages,
            total_state_bits,
            last_pos
        );
    }
    // src params
    let mut read_state_bits = 0;
    for (param, _) in src_params {
        match param {
            InfDataParam::EndPos(p) => {
                let pos = *p / dp_len;
                if last_pos != pos {
                    total_stages += 1; // movement stage
                    test_println!("    Read1: Move to last position: {} {}", last_pos, pos);
                }
                last_pos = pos;
                read_state_bits += 1;
            }
            InfDataParam::TempBuffer(pos) => {
                if last_pos != *pos {
                    total_stages += 1; // movement stage
                    test_println!("    Read2: Move to last position: {} {}", last_pos, *pos);
                }
                last_pos = *pos;
                read_state_bits += dp_len;
            }
            _ => {
                read_state_bits += dp_len;
            }
        }
        total_stages += 2; // read stage and store stage
    }
    test_println!(
        "  Read: TotalStages: {}, ReadStateBits: {}, LastPos: {}",
        total_stages,
        read_state_bits,
        last_pos
    );
    total_stages += 1; // process stage and store results
    test_println!(
        "  Process: TotalStages: {}, ReadStateBits: {}, LastPos: {}",
        total_stages,
        read_state_bits,
        last_pos
    );
    let mut write_state_bits = 0;
    for (param, _) in dests {
        match param {
            InfDataParam::EndPos(p) => {
                total_stages += 1; // read stage for keep values
                let pos = *p / dp_len;
                if last_pos != pos {
                    total_stages += 1; // movement stage
                    test_println!("    Write1: Move to last position: {} {}", last_pos, pos);
                }
                last_pos = pos;
                write_state_bits += 1;
            }
            InfDataParam::TempBuffer(pos) => {
                if last_pos != *pos {
                    total_stages += 1; // movement stage
                    test_println!("    Write2: Move to last position: {} {}", last_pos, *pos);
                }
                last_pos = *pos;
                write_state_bits += dp_len;
            }
            _ => {
                write_state_bits += dp_len;
            }
        }
        total_stages += 1; // write stage
    }
    test_println!(
        "  Write: TotalStages: {}, WriteStateBits: {}, LastPos: {}",
        total_stages,
        write_state_bits,
        last_pos
    );
    // move to next data part
    total_stages += 1;
    test_println!(
        "  EndStage: TotalStages: {}, WriteStateBits: {}, LastPos: {}",
        total_stages,
        write_state_bits,
        last_pos
    );
    // end_stage - stage where is end of algorithm - start moving to start.
    let end_stage = total_stages;
    // add move back stages
    total_stages += 1 + usize::from(use_mem_address) + usize::from(use_proc_id);
    // calculate total state bits
    let total_state_bits = total_state_bits + std::cmp::max(read_state_bits, write_state_bits);
    let total_stages = total_stages;
    test_println!(
        "  Write: TotalStages: {}, TotalStateBits: {}, LastPos: {}",
        total_stages,
        total_state_bits,
        last_pos
    );

    // main routine to generate stages
    let state_start = output_state.bitnum();
    let stage_type_len = calc_log_bits(total_stages);
    extend_output_state(
        state_start,
        stage_type_len + total_state_bits + func.state_len(),
        input,
    );
    let stage = input.state.clone().subvalue(state_start, stage_type_len);
    let state_vars = input
        .state
        .clone()
        .subvalue(state_start + stage_type_len, total_state_bits);
    let func_state = input.state.clone().subvalue(
        state_start + stage_type_len + total_state_bits,
        func.state_len(),
    );

    // start
    let output_base = InfParOutputSys::new(config);
    let create_out_state = |s, sv, fs| output_state.clone().concat(s).concat(sv).concat(fs);
    let mut last_pos = 0;
    let mut outputs = vec![];
    // read src_params and dests end pos
    for (start, list) in [(0, src_params), (src_len, dests)] {
        test_println!("  EndPosList: {:?}", list);
        let mut first = true;
        for (i, (_, end_pos)) in list.into_iter().enumerate() {
            let pos = end_pos / dp_len;
            let mut do_read = false;
            if last_pos != pos {
                // movement stage
                let (output, _) = move_data_pos_stage(
                    create_out_state(
                        UDynVarSys::from_n(outputs.len(), stage_type_len),
                        state_vars.clone(),
                        func_state.clone(),
                    ),
                    create_out_state(
                        UDynVarSys::from_n(outputs.len() + 1, stage_type_len),
                        state_vars.clone(),
                        func_state.clone(),
                    ),
                    input,
                    DKIND_TEMP_BUFFER,
                    if last_pos < pos {
                        DPMOVE_FORWARD
                    } else {
                        DPMOVE_BACKWARD
                    },
                    if last_pos < pos {
                        pos - last_pos
                    } else {
                        last_pos - pos
                    } as u64,
                );
                test_println!(
                    "    GenEndPos: {} {} {}: Move to last position: {} {}: {} {}",
                    i,
                    end_pos,
                    outputs.len(),
                    last_pos,
                    pos,
                    if last_pos < pos {
                        DPMOVE_FORWARD
                    } else {
                        DPMOVE_BACKWARD
                    },
                    if last_pos < pos {
                        pos - last_pos
                    } else {
                        last_pos - pos
                    } as u64,
                );
                outputs.push(output);
                do_read = true;
            } else if first {
                do_read = true;
            }
            // read stage
            if do_read {
                // read stage
                let mut output = output_base.clone();
                output.state = create_out_state(
                    UDynVarSys::from_n(outputs.len() + 1, stage_type_len),
                    state_vars.clone(),
                    func_state.clone(),
                );
                output.dkind = DKIND_TEMP_BUFFER.into();
                output.dpr = true.into();
                test_println!(
                    "    GenEndPos {} {} {}: Read stage: {} {}",
                    i,
                    end_pos,
                    outputs.len(),
                    last_pos,
                    pos
                );
                outputs.push(output);
                // store stage
                let mut output = output_base.clone();
                let end_poses = UDynVarSys::from_iter(
                    list[i..]
                        .into_iter()
                        .take_while(|(_, end_pos)| {
                            // while end_pos is same data part
                            pos == (end_pos / dp_len)
                        })
                        .enumerate()
                        .map(|(x, (_, end_pos))| {
                            state_vars.bit(start + i + x) | input.dpval.bit(end_pos % dp_len)
                        }),
                );
                test_println!(
                    "    GenEndPos {} {} {}: Write stage: {} {}: {:?}",
                    i,
                    end_pos,
                    outputs.len(),
                    last_pos,
                    pos,
                    list[i..]
                        .into_iter()
                        .take_while(|(_, end_pos)| {
                            // while end_pos is same data part
                            pos == (end_pos / dp_len)
                        })
                        .enumerate()
                        .collect::<Vec<_>>()
                );
                let new_state_vars = UDynVarSys::from_iter((0..total_state_bits).map(|x| {
                    if x < start + i || x >= start + i + end_poses.len() {
                        // old bit
                        state_vars.bit(x)
                    } else {
                        // new bit from end pos
                        end_poses.bit(x - i - start).clone()
                    }
                }));
                output.state = create_out_state(
                    UDynVarSys::from_n(outputs.len() + 1, stage_type_len),
                    new_state_vars,
                    func_state.clone(),
                );
                outputs.push(output);
            }
            last_pos = pos;
            first = false;
        }
    }
    // read params - read phase
    let mut state_pos = src_len + dest_len;
    test_println!("  GenRead");
    for (param, _) in src_params {
        let pos = match param {
            InfDataParam::EndPos(p) => Some(*p / dp_len),
            InfDataParam::TempBuffer(pos) => Some(*pos),
            _ => None,
        };
        if let Some(pos) = pos {
            if last_pos != pos {
                // movement stage
                let (output, _) = move_data_pos_stage(
                    create_out_state(
                        UDynVarSys::from_n(outputs.len(), stage_type_len),
                        state_vars.clone(),
                        func_state.clone(),
                    ),
                    create_out_state(
                        UDynVarSys::from_n(outputs.len() + 1, stage_type_len),
                        state_vars.clone(),
                        func_state.clone(),
                    ),
                    input,
                    DKIND_TEMP_BUFFER,
                    if last_pos < pos {
                        DPMOVE_FORWARD
                    } else {
                        DPMOVE_BACKWARD
                    },
                    if last_pos < pos {
                        pos - last_pos
                    } else {
                        last_pos - pos
                    } as u64,
                );
                test_println!(
                    "    GenRead {:?} {}: Move to last position: {} {}: {} {}",
                    param,
                    outputs.len(),
                    last_pos,
                    pos,
                    if last_pos < pos {
                        DPMOVE_FORWARD
                    } else {
                        DPMOVE_BACKWARD
                    },
                    if last_pos < pos {
                        pos - last_pos
                    } else {
                        last_pos - pos
                    } as u64,
                );
                outputs.push(output);
                last_pos = pos;
            }
        }
        // read stage
        let mut output = output_base.clone();
        output.state = create_out_state(
            UDynVarSys::from_n(outputs.len() + 1, stage_type_len),
            state_vars.clone(),
            func_state.clone(),
        );
        output.dkind = match param {
            InfDataParam::MemAddress => DKIND_MEM_ADDRESS,
            InfDataParam::ProcId => DKIND_PROC_ID,
            InfDataParam::TempBuffer(_) | InfDataParam::EndPos(_) => DKIND_TEMP_BUFFER,
        }
        .into();
        output.dpr = true.into();
        output.dpmove = if (!use_write_mem_address && *param == InfDataParam::MemAddress)
            || *param == InfDataParam::ProcId
        {
            // move forward proc_id or mem_address and mem_address not used to write.
            DPMOVE_FORWARD
        } else {
            DPMOVE_NOTHING
        }
        .into();
        test_println!(
            "    GenRead {:?} {}: Read stage {}: StatePos: {}, DPMove: {}",
            param,
            outputs.len(),
            last_pos,
            state_pos,
            (!use_write_mem_address && *param == InfDataParam::MemAddress)
                || *param == InfDataParam::ProcId
        );
        outputs.push(output);
        // store stage
        let param_len = if matches!(*param, InfDataParam::EndPos(_)) {
            1
        } else {
            dp_len
        };
        let mut output = output_base.clone();
        let new_state_vars = UDynVarSys::from_iter((0..total_state_bits).map(|x| {
            if x < state_pos || x >= state_pos + param_len {
                state_vars.bit(x)
            } else {
                input.dpval.bit(x - state_pos)
            }
        }));
        output.state = create_out_state(
            UDynVarSys::from_n(outputs.len() + 1, stage_type_len),
            new_state_vars,
            func_state.clone(),
        );
        test_println!(
            "    GenRead {:?} {}: Store stage {}: StatePos: {}, ParamLen: {}",
            param,
            outputs.len(),
            last_pos,
            state_pos,
            param_len
        );
        outputs.push(output);
        state_pos += param_len;
    }
    // process stage
    let func_inputs = {
        let mut func_inputs = vec![];
        let mut state_pos = src_len + dest_len;
        for (i, (param, _)) in src_params.iter().enumerate() {
            let param_len = if matches!(*param, InfDataParam::EndPos(_)) {
                1
            } else {
                dp_len
            };
            func_inputs.push(dynint_ite(
                // use src end_pos to filter function inputs (if 1 then zeroing)
                !state_vars.bit(i),
                UDynVarSys::from_iter((0..param_len).map(|x| state_vars.bit(state_pos + x))),
                UDynVarSys::from_n(0u8, param_len),
            ));
            println!(
                "  FuncInputs: {} {:?}: StatePos: {}, ParamLen: {}",
                i, src_params, state_pos, param_len
            );
            state_pos += param_len;
        }
        func_inputs
    };
    let (next_func_state, outvals) = func.output(func_state.clone(), &func_inputs);
    let mut output = output_base.clone();
    // get function output bitvector
    let func_outputs = {
        let mut func_output_bits = vec![];
        for ((param, _), outval) in dests.into_iter().zip(outvals.into_iter()) {
            let param_len = if matches!(*param, InfDataParam::EndPos(_)) {
                1
            } else {
                dp_len
            };
            func_output_bits.extend((0..param_len).map(|x| outval.bit(x)));
            println!("  FuncOutputs: {:?}: ParamLen: {}", dests, param_len);
        }
        if read_state_bits > write_state_bits {
            // fix length of func output bits - fix if read state bits is longer
            // than write state bits
            func_output_bits
                .extend((write_state_bits..read_state_bits).map(|_| BoolVarSys::from(false)));
        }
        UDynVarSys::from_iter(func_output_bits)
    };
    let next_stage = dynint_ite(
        // AND for all dest end_pos: E0 and E1 and E2 ... EN. If 1 then go to end.
        (src_len..src_len + dest_len)
            .fold(BoolVarSys::from(true), |a, x| a.clone() & state_vars.bit(x)),
        UDynVarSys::from_n(end_stage, stage_type_len),
        UDynVarSys::from_n(outputs.len() + 1, stage_type_len),
    );
    println!(
        "  NextStege: {}..{}: EndStage: {}",
        src_len,
        src_len + dest_len,
        end_stage
    );
    // outputs start at same position as inputs
    let state_pos = src_len + dest_len;
    assert_eq!(state_pos + func_outputs.bitnum(), total_state_bits);
    output.state = create_out_state(
        next_stage,
        state_vars
            .clone()
            .subvalue(0, state_pos)
            .concat(func_outputs),
        next_func_state,
    );
    outputs.push(output);

    // start from same position in states as read phase.
    let mut state_pos = src_len + dest_len;
    test_println!("  GenWrite");
    // write stages - write phase
    for (i, (param, _)) in dests.into_iter().enumerate() {
        let pos = match param {
            InfDataParam::EndPos(p) => Some(*p / dp_len),
            InfDataParam::TempBuffer(pos) => Some(*pos),
            _ => None,
        };
        if let Some(pos) = pos {
            if last_pos != pos {
                // movement stage
                let (output, _) = move_data_pos_stage(
                    create_out_state(
                        UDynVarSys::from_n(outputs.len(), stage_type_len),
                        state_vars.clone(),
                        func_state.clone(),
                    ),
                    create_out_state(
                        UDynVarSys::from_n(outputs.len() + 1, stage_type_len),
                        state_vars.clone(),
                        func_state.clone(),
                    ),
                    input,
                    DKIND_TEMP_BUFFER,
                    if last_pos < pos {
                        DPMOVE_FORWARD
                    } else {
                        DPMOVE_BACKWARD
                    },
                    if last_pos < pos {
                        pos - last_pos
                    } else {
                        last_pos - pos
                    } as u64,
                );
                test_println!(
                    "    GenWrite {} {:?} {}: Move to last position: {} {}: {} {}",
                    i,
                    param,
                    outputs.len(),
                    last_pos,
                    pos,
                    if last_pos < pos {
                        DPMOVE_FORWARD
                    } else {
                        DPMOVE_BACKWARD
                    },
                    if last_pos < pos {
                        pos - last_pos
                    } else {
                        last_pos - pos
                    } as u64,
                );
                outputs.push(output);
                last_pos = pos;
            }
        }
        match param {
            InfDataParam::EndPos(p) => {
                // read stage
                let mut output = output_base.clone();
                output.state = create_out_state(
                    UDynVarSys::from_n(outputs.len() + 1, stage_type_len),
                    state_vars.clone(),
                    func_state.clone(),
                );
                output.dkind = DKIND_TEMP_BUFFER.into();
                output.dpr = true.into();
                test_println!(
                    "    GenWrite {:?} {}: Read stage: {}: StatePos: {}",
                    param,
                    outputs.len(),
                    last_pos,
                    state_pos,
                );
                outputs.push(output);
                // write stage
                let mut output = output_base.clone();
                output.state = create_out_state(
                    UDynVarSys::from_n(outputs.len() + 1, stage_type_len),
                    state_vars.clone(),
                    func_state.clone(),
                );
                output.dkind = DKIND_TEMP_BUFFER.into();
                // use dest end pos to write
                output.dpw = !state_vars.bit(src_len + i);
                let bit = p % dp_len;
                output.dpval = UDynVarSys::from_iter((0..dp_len).map(|x| {
                    if bit == x {
                        // new value
                        state_vars.bit(state_pos)
                    } else {
                        // keep old value
                        input.dpval.bit(x)
                    }
                }));
                outputs.push(output);
                test_println!(
                    "    GenWrite {:?} {}: Write stage: {}: StatePos: {}, DPW: {}",
                    param,
                    outputs.len(),
                    last_pos,
                    state_pos,
                    src_len + i
                );
                state_pos += 1;
            }
            InfDataParam::MemAddress | InfDataParam::TempBuffer(_) => {
                // write stage
                let mut output = output_base.clone();
                output.state = create_out_state(
                    UDynVarSys::from_n(outputs.len() + 1, stage_type_len),
                    state_vars.clone(),
                    func_state.clone(),
                );
                output.dkind = if *param == InfDataParam::MemAddress {
                    DKIND_MEM_ADDRESS
                } else {
                    DKIND_TEMP_BUFFER
                }
                .into();
                // use dest end pos to write
                output.dpw = !state_vars.bit(src_len + i);
                if *param == InfDataParam::MemAddress && use_write_mem_address {
                    // move forward mem address if writing
                    output.dpmove = DPMOVE_FORWARD.into();
                }
                output.dpval =
                    UDynVarSys::from_iter((0..dp_len).map(|x| state_vars.bit(state_pos + x)));
                outputs.push(output);
                test_println!(
                    "    GenWrite {:?} {}: Write stage: {}: StatePos: {}, DPW: {}, DPMove: {}",
                    param,
                    outputs.len(),
                    last_pos,
                    state_pos,
                    src_len + i,
                    *param == InfDataParam::MemAddress && use_write_mem_address,
                );
                state_pos += dp_len;
            }
            _ => {
                panic!("Unexpected!");
            }
        }
    }
    // stage to move to next data part
    // movement stage
    let (output, _) = move_data_pos_stage(
        create_out_state(
            UDynVarSys::from_n(outputs.len(), stage_type_len),
            state_vars.clone(),
            func_state.clone(),
        ),
        create_out_state(
            // move to start
            UDynVarSys::from_n(0u8, stage_type_len),
            state_vars.clone(),
            func_state.clone(),
        ),
        input,
        DKIND_TEMP_BUFFER,
        DPMOVE_FORWARD,
        ((temp_buffer_step as usize) - last_pos) as u64,
    );
    test_println!("  GenToNext: {} {}", last_pos, outputs.len());
    outputs.push(output);

    // end phase move back
    let (output, mut end_of_stage_final) = data_pos_to_start_stage(
        create_out_state(
            UDynVarSys::from_n(outputs.len(), stage_type_len),
            state_vars.clone(),
            func_state.clone(),
        ),
        create_out_state(
            if use_mem_address || use_proc_id {
                UDynVarSys::from_n(outputs.len() + 1, stage_type_len)
            } else {
                UDynVarSys::from_n(0u8, stage_type_len)
            },
            state_vars.clone(),
            func_state.clone(),
        ),
        input,
        DKIND_TEMP_BUFFER,
    );
    outputs.push(output);
    test_println!("  MoveToStartTempBuffer");
    if use_mem_address {
        let (output, end_of_stage) = data_pos_to_start_stage(
            create_out_state(
                UDynVarSys::from_n(outputs.len(), stage_type_len),
                state_vars.clone(),
                func_state.clone(),
            ),
            create_out_state(
                if use_proc_id {
                    UDynVarSys::from_n(outputs.len() + 1, stage_type_len)
                } else {
                    UDynVarSys::from_n(0u8, stage_type_len)
                },
                state_vars.clone(),
                func_state.clone(),
            ),
            input,
            DKIND_MEM_ADDRESS,
        );
        outputs.push(output);
        test_println!("  MoveToStartMemAddress");
        // this is end of stage
        end_of_stage_final = end_of_stage;
    }
    if use_proc_id {
        let (output, end_of_stage) = data_pos_to_start_stage(
            create_out_state(
                UDynVarSys::from_n(outputs.len(), stage_type_len),
                state_vars.clone(),
                func_state.clone(),
            ),
            create_out_state(
                UDynVarSys::from_n(0u8, stage_type_len),
                state_vars.clone(),
                func_state.clone(),
            ),
            input,
            DKIND_PROC_ID,
        );
        outputs.push(output);
        // this is end of stage
        end_of_stage_final = end_of_stage;
        test_println!("  MoveToStartProcId");
    }
    test_println!("  OutputsLen: {}", outputs.len());
    assert_eq!(total_stages, outputs.len());
    // prepare end bit
    let end = (&stage).equal(total_stages - 1) & end_of_stage_final;
    // finish generation
    finish_stage_with_table(output_state, next_state, input, outputs, stage, end)
}
