use gategen::boolvar::*;
use gategen::dynintvar::*;
use gategen::intvar::*;
use infmachine_gen::*;

use std::collections::{BinaryHeap, HashMap};
use std::fmt::Debug;
use std::hash::Hash;

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
    fn output(
        &self,
        input_state: UDynVarSys,
        inputs: &[UDynVarSys],
    ) -> (UDynVarSys, Vec<UDynVarSys>);
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
    par_process_proc_id_to_temp_buffer_stage(
        output_state,
        next_state,
        input,
        temp_buffer_step,
        temp_buffer_step_pos,
        Copy1Func::new(),
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
    par_process_proc_id_to_mem_address_stage(
        output_state,
        next_state,
        input,
        temp_buffer_step,
        Copy1Func::new(),
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
    par_process_temp_buffer_to_mem_address_stage(
        output_state,
        next_state,
        input,
        temp_buffer_step,
        temp_buffer_step_pos,
        Copy1Func::new(),
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
    par_process_mem_address_to_temp_buffer_stage(
        output_state,
        next_state,
        input,
        temp_buffer_step,
        temp_buffer_step_pos,
        Copy1Func::new(),
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
    assert_eq!(output_state.bitnum(), next_state.bitnum());
    assert_ne!(temp_buffer_step, 0);
    assert_ne!(tbs_src_pos, 0);
    assert_ne!(tbs_dest_pos, 0);
    assert_ne!(tbs_src_pos, tbs_dest_pos);
    assert!(tbs_src_pos < temp_buffer_step);
    assert!(tbs_dest_pos < temp_buffer_step);
    let config = input.config();
    let dp_len = config.data_part_len as usize;
    let state_start = output_state.bitnum();
    type StageType = U4VarSys;
    extend_output_state(state_start, StageType::BITS + dp_len, input);
    let stage =
        StageType::try_from(input.state.clone().subvalue(state_start, StageType::BITS)).unwrap();
    let value = input
        .state
        .clone()
        .subvalue(state_start + StageType::BITS, dp_len);
    // start
    let output_base = InfParOutputSys::new(config);
    let create_out_state = |s: StageType, v| output_state.clone().concat(s.into()).concat(v);
    let value_zero = UDynVarSys::from_n(0u8, dp_len);
    // Algorithm:
    // 0. If data_part_len == 1: Make forward temp_buffer_pos to move to proc_id end marker.
    // tidx - stage index for main routine
    let (tidx, tbs_src_pos, tbs_dest_pos) = if dp_len <= 1 {
        assert!(temp_buffer_step >= 2);
        assert!(tbs_src_pos >= 2);
        assert!(tbs_dest_pos >= 2);
        if proc_id_end_pos {
            (1u8, tbs_src_pos - 1, tbs_dest_pos - 1)
        } else {
            (0u8, tbs_src_pos, tbs_dest_pos)
        }
    } else {
        (0u8, tbs_src_pos, tbs_dest_pos)
    };
    // make temp buffer position to 1.
    let mut output_tshift = output_base.clone();
    if dp_len == 1 && proc_id_end_pos {
        output_tshift.state = create_out_state(StageType::from(1u8), value_zero.clone());
        output_tshift.dpmove = U2VarSys::from(DPMOVE_FORWARD);
        output_tshift.dkind = DKIND_TEMP_BUFFER.into();
    }
    // 0: 1. Load temp_buffer data part.
    let mut output_0 = output_base.clone();
    output_0.state = create_out_state(StageType::from(tidx + 1u8), value_zero.clone());
    output_0.dkind = DKIND_TEMP_BUFFER.into();
    output_0.dpr = true.into();
    // 1: 2. If data_part==0: then:
    let no_end_pos = if dp_len == 1 {
        !(&input.dpval).bit(0)
    } else {
        if proc_id_end_pos {
            !(&input.dpval).bit(1)
        } else {
            !(&input.dpval).bit(0)
        }
    };
    let mut output_1 = output_base.clone();
    output_1.state = create_out_state(
        int_ite(
            no_end_pos,
            StageType::from(tidx + 2u8),
            // go to 9.
            StageType::from(tidx + 8u8),
        ),
        value_zero.clone(),
    );
    // 2: 3. Move temp buffer position forward by tbs_src_pos.
    let (output_2, _) = move_data_pos_stage(
        create_out_state(stage.clone(), value_zero.clone()),
        create_out_state(StageType::from(tidx + 3u8), value_zero.clone()),
        input,
        DKIND_TEMP_BUFFER,
        DPMOVE_FORWARD,
        tbs_src_pos as u64,
    );
    // 3: 4. Load temp buffer data_part from tbs_src_pos.
    let mut output_3 = output_base.clone();
    output_3.state = create_out_state(StageType::from(tidx + 4u8), value_zero.clone());
    output_3.dkind = DKIND_TEMP_BUFFER.into();
    output_3.dpr = true.into();
    // 4: 5. Store data part into state value
    let mut output_4 = output_base.clone();
    output_4.state = create_out_state(StageType::from(tidx + 5u8), input.dpval.clone());
    // 5: 6. Move temp_buffer position forward to tbs_dest_pos.
    let (output_5, _) = move_data_pos_stage(
        create_out_state(stage.clone(), value.clone()),
        create_out_state(StageType::from(tidx + 6u8), value.clone()),
        input,
        DKIND_TEMP_BUFFER,
        if tbs_src_pos < tbs_dest_pos {
            DPMOVE_FORWARD
        } else {
            DPMOVE_BACKWARD
        },
        if tbs_src_pos < tbs_dest_pos {
            (tbs_dest_pos - tbs_src_pos) as u64
        } else {
            (tbs_src_pos - tbs_dest_pos) as u64
        },
    );
    // 6: 7. Store value into destination temp buffer position.
    let mut output_6 = output_base.clone();
    output_6.state = create_out_state(StageType::from(tidx + 7u8), value_zero.clone());
    output_6.dkind = DKIND_TEMP_BUFFER.into();
    output_6.dpw = true.into();
    output_6.dpval = value.clone();
    // 7: 8. Move temp buffer position forward by (temp_buffer_step - tbs_dest_pos)
    // 8.1. Go to 1.
    let (output_7, _) = move_data_pos_stage(
        create_out_state(stage.clone(), value_zero.clone()),
        create_out_state(StageType::from(tidx), value_zero.clone()),
        input,
        DKIND_TEMP_BUFFER,
        DPMOVE_FORWARD,
        (temp_buffer_step - tbs_dest_pos) as u64,
    );
    // 9. Else (step 1)
    // 8: 10. Move temp buffer position to start.
    let (output_8, end_8) = data_pos_to_start_stage(
        create_out_state(stage.clone(), value_zero.clone()),
        create_out_state(StageType::from(tidx), value_zero.clone()),
        input,
        DKIND_TEMP_BUFFER,
    );
    // 11. End of algorithm.
    let end = end_8 & (&stage).equal(tidx + 8u8);
    // finishing
    let mut output_stages = vec![
        output_0, output_1, output_2, output_3, output_4, output_5, output_6, output_7, output_8,
    ];
    if dp_len == 1 && proc_id_end_pos {
        output_stages.insert(0, output_tshift);
    }
    finish_stage_with_table(
        output_state,
        next_state,
        input,
        output_stages,
        stage.into(),
        end,
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
    assert_eq!(output_state.bitnum(), next_state.bitnum());
    assert_ne!(temp_buffer_step, 0);
    assert_ne!(temp_buffer_step_pos, 0);
    assert!(temp_buffer_step_pos < temp_buffer_step);
    let config = input.config();
    let dp_len = config.data_part_len as usize;
    let state_start = output_state.bitnum();
    type StageType = U4VarSys;
    extend_output_state(state_start, StageType::BITS + func.state_len(), input);
    let stage =
        StageType::try_from(input.state.clone().subvalue(state_start, StageType::BITS)).unwrap();
    let func_state = input
        .state
        .clone()
        .subvalue(state_start + StageType::BITS, func.state_len());
    // start
    let output_base = InfParOutputSys::new(config);
    let create_out_state = |s: StageType, fs| output_state.clone().concat(s.into()).concat(fs);
    // Algorithm:
    // 0. If data_part_len == 1: Make forward temp_buffer_pos to move to proc_id end marker.
    // tidx - stage index for main routine
    let (tidx, temp_buffer_step_pos) = if config.data_part_len <= 1 {
        assert!(temp_buffer_step >= 2);
        assert!(temp_buffer_step_pos >= 2);
        (1u8, temp_buffer_step_pos - 1)
    } else {
        (0u8, temp_buffer_step_pos)
    };
    // make temp buffer position to 1.
    let mut output_tshift = output_base.clone();
    if config.data_part_len == 1 {
        output_tshift.state = create_out_state(StageType::from(1u8), func_state.clone());
        output_tshift.dpmove = U2VarSys::from(DPMOVE_FORWARD);
        output_tshift.dkind = DKIND_TEMP_BUFFER.into();
    }
    // 0: 1. Load temp_buffer data part.
    let mut output_0 = output_base.clone();
    output_0.state = create_out_state(StageType::from(tidx + 1u8), func_state.clone());
    output_0.dkind = DKIND_TEMP_BUFFER.into();
    output_0.dpr = true.into();
    // 1: 2. If data_part==0: then:
    let mut output_1 = output_base.clone();
    let no_end_of_proc_id = if dp_len >= 2 {
        !(&input.dpval).bit(1)
    } else {
        !(&input.dpval).bit(0)
    };
    output_1.state = create_out_state(
        int_ite(
            no_end_of_proc_id,
            StageType::from(tidx + 2u8),
            // go to 9.
            StageType::from(tidx + 6u8),
        ),
        func_state.clone(),
    );
    // 2: 3. Move temp buffer position forward by temp_buffer_step_pos.
    let (output_2, _) = move_data_pos_stage(
        create_out_state(stage.clone(), func_state.clone()),
        create_out_state(StageType::from(tidx + 3u8), func_state.clone()),
        input,
        DKIND_TEMP_BUFFER,
        DPMOVE_FORWARD,
        temp_buffer_step_pos as u64,
    );
    // 3: 4. Load proc_id data_part.
    // 5. Move forward proc id position.
    let mut output_3 = output_base.clone();
    output_3.state = create_out_state(StageType::from(tidx + 4u8), func_state.clone());
    output_3.dkind = DKIND_PROC_ID.into();
    output_3.dpr = true.into();
    output_3.dpmove = DPMOVE_FORWARD.into();
    // 4: 6. Process data part and store result into current temp buffer position.
    let (out_func_state, out_value) = func.output(func_state.clone(), input.dpval.clone());
    let mut output_4 = output_base.clone();
    output_4.state = create_out_state(StageType::from(tidx + 5u8), out_func_state);
    output_4.dkind = DKIND_TEMP_BUFFER.into();
    output_4.dpw = true.into();
    output_4.dpval = out_value;
    // 5: 7. Move temp_buffer position forward by (temp_buffer_step - temp_buffer_step_pos).
    // 5: 8. Go to 1.
    let (output_5, _) = move_data_pos_stage(
        create_out_state(stage.clone(), func_state.clone()),
        create_out_state(StageType::from(tidx), func_state.clone()),
        input,
        DKIND_TEMP_BUFFER,
        DPMOVE_FORWARD,
        (temp_buffer_step - temp_buffer_step_pos) as u64,
    );
    // 9. Else (step 1)
    // 6: 10. Move temp buffer position to start.
    let (output_6, _) = data_pos_to_start_stage(
        create_out_state(stage.clone(), func_state.clone()),
        create_out_state(StageType::from(tidx + 7u8), func_state.clone()),
        input,
        DKIND_TEMP_BUFFER,
    );
    // 7: 11. Move proc id position to start.
    let (output_7, end_7) = data_pos_to_start_stage(
        create_out_state(stage.clone(), func_state.clone()),
        create_out_state(StageType::from(0u8), func_state.clone()),
        input,
        DKIND_PROC_ID,
    );
    // 12. End of algorithm.
    let end = end_7 & (&stage).equal(tidx + 7u8);
    // finishing
    let mut output_stages = vec![
        output_0, output_1, output_2, output_3, output_4, output_5, output_6, output_7,
    ];
    if config.data_part_len <= 1 {
        output_stages.insert(0, output_tshift);
    }
    finish_stage_with_table(
        output_state,
        next_state,
        input,
        output_stages,
        stage.into(),
        end,
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
    assert_eq!(output_state.bitnum(), next_state.bitnum());
    assert_ne!(temp_buffer_step, 0);
    let config = input.config();
    let dp_len = config.data_part_len as usize;
    let state_start = output_state.bitnum();
    type StageType = U4VarSys;
    extend_output_state(state_start, StageType::BITS + 1 + func.state_len(), input);
    let stage =
        StageType::try_from(input.state.clone().subvalue(state_start, StageType::BITS)).unwrap();
    // dp_zero - if 1 - then zero data part stored to memory address
    let dp_zero = input.state.bit(state_start + StageType::BITS);
    let func_state = input
        .state
        .clone()
        .subvalue(state_start + StageType::BITS + 1, func.state_len());
    // start
    let output_base = InfParOutputSys::new(config);
    let create_out_state = |s: StageType, dpz, fs| {
        output_state
            .clone()
            .concat(s.into())
            .concat(UDynVarSys::filled(1, dpz))
            .concat(fs)
    };
    // temp_buffer_step_fixed - fixed if temp buffer position is 1 (dp_len=1).
    let temp_buffer_step_fixed = if config.data_part_len <= 1 {
        assert!(temp_buffer_step >= 2);
        temp_buffer_step - 1
    } else {
        temp_buffer_step
    };
    // Algorithm:
    // 0: 1. Load temp_buffer data part.
    let mut output_0 = output_base.clone();
    output_0.state = create_out_state(StageType::from(1u8), dp_zero.clone(), func_state.clone());
    output_0.dkind = DKIND_TEMP_BUFFER.into();
    output_0.dpr = true.into();
    if config.data_part_len <= 1 {
        // move forward temp buffer position to get proc_id_end_pos
        output_0.dpmove = DPMOVE_FORWARD.into();
    }
    // 1: 2. If data_part==0: then:
    let (output_1, output_1_1, tidx) = if config.data_part_len <= 1 {
        // make two steps because data_part have 1 bit
        let mut output_1 = output_base.clone();
        output_1.state = create_out_state(
            int_ite(
                !(&input.dpval).bit(0),
                StageType::from(2u8),
                // go to 9.
                StageType::from(1 + 5u8),
            ),
            dp_zero.clone(),
            func_state.clone(),
        );
        // load proc_id position end pos and update dp_zero
        output_1.dkind = DKIND_TEMP_BUFFER.into();
        output_1.dpr = true.into();
        // next stage
        let mut output_1_1 = output_base.clone();
        output_1_1.state = create_out_state(
            StageType::from(1 + 2u8),
            // update dp_zero: by joining with proc_id_end_pos marker
            &dp_zero | input.dpval.bit(0),
            func_state.clone(),
        );
        (output_1, output_1_1, 1)
    } else {
        // normal if data_part can hold two bits
        let mut output_1 = output_base.clone();
        output_1.state = create_out_state(
            int_ite(
                !(&input.dpval).bit(0),
                StageType::from(2u8),
                // go to 9.
                StageType::from(5u8),
            ),
            // update dp_zero: by joining with proc_id_end_pos marker
            &dp_zero | input.dpval.bit(1),
            func_state.clone(),
        );
        (output_1.clone(), output_1, 0)
    };
    // 2: 3. Load proc_id data_part.
    // 2: 4. Move forward proc id position.
    let mut output_2 = output_base.clone();
    output_2.state = create_out_state(
        StageType::from(tidx + 3u8),
        dp_zero.clone(),
        func_state.clone(),
    );
    output_2.dkind = DKIND_PROC_ID.into();
    output_2.dpr = true.into();
    output_2.dpmove = DPMOVE_FORWARD.into();
    // 3: 4. Store data part into current temp buffer position.
    // 3: 5. Move mem_address position forward.
    let (out_func_state, out_value) = func.output(
        func_state.clone(),
        dynint_ite(
            dp_zero.clone(),
            UDynVarSys::from_n(0u8, dp_len),
            input.dpval.clone(),
        ),
    );
    let mut output_3 = output_base.clone();
    output_3.state = create_out_state(StageType::from(tidx + 4u8), dp_zero.clone(), out_func_state);
    output_3.dkind = DKIND_MEM_ADDRESS.into();
    output_3.dpw = true.into();
    output_3.dpval = out_value;
    output_3.dpmove = DPMOVE_FORWARD.into();
    // 4: 7. Move temp_buffer position forward by temp_buffer_step.
    // 4: 8. Go to 1.
    let (output_4, _) = move_data_pos_stage(
        create_out_state(stage.clone(), dp_zero.clone(), func_state.clone()),
        create_out_state(StageType::from(0u8), dp_zero.clone(), func_state.clone()),
        input,
        DKIND_TEMP_BUFFER,
        DPMOVE_FORWARD,
        temp_buffer_step_fixed as u64,
    );
    // 9. Else (step 1)
    // 5: 10. Move mem address position to start.
    let (output_5, _) = data_pos_to_start_stage(
        create_out_state(stage.clone(), dp_zero.clone(), func_state.clone()),
        create_out_state(
            StageType::from(tidx + 6u8),
            dp_zero.clone(),
            func_state.clone(),
        ),
        input,
        DKIND_MEM_ADDRESS,
    );
    // 6: 11. Move temp buffer position to start.
    let (output_6, _) = data_pos_to_start_stage(
        create_out_state(stage.clone(), dp_zero.clone(), func_state.clone()),
        create_out_state(
            StageType::from(tidx + 7u8),
            dp_zero.clone(),
            func_state.clone(),
        ),
        input,
        DKIND_TEMP_BUFFER,
    );
    // 7: 12. Move proc id position to start.
    let (output_7, end_7) = data_pos_to_start_stage(
        create_out_state(stage.clone(), dp_zero.clone(), func_state.clone()),
        create_out_state(StageType::from(0u8), dp_zero.clone(), func_state.clone()),
        input,
        DKIND_PROC_ID,
    );
    // 13. End of algorithm.
    let end = end_7 & (&stage).equal(tidx + 7u8);
    // finishing
    let mut output_stages = vec![
        output_0, output_1, output_2, output_3, output_4, output_5, output_6, output_7,
    ];
    if config.data_part_len <= 1 {
        // insert additional stage to routine
        output_stages.insert(2, output_1_1);
    }
    finish_stage_with_table(
        output_state,
        next_state,
        input,
        output_stages,
        stage.into(),
        end,
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
    assert_eq!(output_state.bitnum(), next_state.bitnum());
    assert_ne!(temp_buffer_step, 0);
    assert_ne!(temp_buffer_step_pos, 0);
    assert!(temp_buffer_step_pos < temp_buffer_step);
    let config = input.config();
    let state_start = output_state.bitnum();
    type StageType = U3VarSys;
    extend_output_state(state_start, StageType::BITS + func.state_len(), input);
    let stage =
        StageType::try_from(input.state.clone().subvalue(state_start, StageType::BITS)).unwrap();
    let func_state = input
        .state
        .clone()
        .subvalue(state_start + StageType::BITS, func.state_len());
    // start
    let output_base = InfParOutputSys::new(config);
    let create_out_state = |s: StageType, fs| output_state.clone().concat(s.into()).concat(fs);
    if config.data_part_len <= 1 {
        assert!(temp_buffer_step >= 2);
        assert!(temp_buffer_step_pos >= 2);
    };
    // Algorithm:
    // 0: 1. Load temp_buffer data part.
    let mut output_0 = output_base.clone();
    output_0.state = create_out_state(StageType::from(1u8), func_state.clone());
    output_0.dkind = DKIND_TEMP_BUFFER.into();
    output_0.dpr = true.into();
    // 1: 2. If data_part==0: then:
    let mut output_1 = output_base.clone();
    output_1.state = create_out_state(
        int_ite(
            !(&input.dpval).bit(0),
            StageType::from(2u8),
            // go to 9.
            StageType::from(6u8),
        ),
        func_state.clone(),
    );
    // 2: 3. Move temp buffer position forward by temp_buffer_step_pos.
    let (output_2, _) = move_data_pos_stage(
        create_out_state(stage.clone(), func_state.clone()),
        create_out_state(StageType::from(3u8), func_state.clone()),
        input,
        DKIND_TEMP_BUFFER,
        DPMOVE_FORWARD,
        temp_buffer_step_pos as u64,
    );
    // 3: 4. Load temp_buffer data_part.
    let mut output_3 = output_base.clone();
    output_3.state = create_out_state(StageType::from(4u8), func_state.clone());
    output_3.dkind = DKIND_TEMP_BUFFER.into();
    output_3.dpr = true.into();
    // 4: 5. Process data part into current mem_address position.
    // 6. Move forward mem_address position.
    let (out_func_state, out_value) = func.output(func_state.clone(), input.dpval.clone());
    let mut output_4 = output_base.clone();
    output_4.state = create_out_state(StageType::from(5u8), out_func_state);
    output_4.dkind = DKIND_MEM_ADDRESS.into();
    output_4.dpw = true.into();
    output_4.dpval = out_value;
    output_4.dpmove = DPMOVE_FORWARD.into();
    // 5: 7. Move temp_buffer position forward by (temp_buffer_step - temp_buffer_step_pos).
    // 5: 8. Go to 1.
    let (output_5, _) = move_data_pos_stage(
        create_out_state(stage.clone(), func_state.clone()),
        create_out_state(StageType::from(0u8), func_state.clone()),
        input,
        DKIND_TEMP_BUFFER,
        DPMOVE_FORWARD,
        (temp_buffer_step - temp_buffer_step_pos) as u64,
    );
    // 9. Else (step 1)
    // 6: 10. Move temp buffer position to start.
    let (output_6, _) = data_pos_to_start_stage(
        create_out_state(stage.clone(), func_state.clone()),
        create_out_state(StageType::from(7u8), func_state.clone()),
        input,
        DKIND_TEMP_BUFFER,
    );
    // 7: 11. Move mem_address position to start.
    let (output_7, end_7) = data_pos_to_start_stage(
        create_out_state(stage.clone(), func_state.clone()),
        create_out_state(StageType::from(0u8), func_state.clone()),
        input,
        DKIND_MEM_ADDRESS,
    );
    // 12. End of algorithm.
    let end = end_7 & (&stage).equal(7u8);
    // finishing
    let output_stages = vec![
        output_0, output_1, output_2, output_3, output_4, output_5, output_6, output_7,
    ];
    finish_stage_with_table(
        output_state,
        next_state,
        input,
        output_stages,
        stage.into(),
        end,
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
    assert_eq!(output_state.bitnum(), next_state.bitnum());
    assert_ne!(temp_buffer_step, 0);
    assert_ne!(temp_buffer_step_pos, 0);
    assert!(temp_buffer_step_pos < temp_buffer_step);
    let config = input.config();
    let state_start = output_state.bitnum();
    type StageType = U3VarSys;
    extend_output_state(state_start, StageType::BITS + func.state_len(), input);
    let stage =
        StageType::try_from(input.state.clone().subvalue(state_start, StageType::BITS)).unwrap();
    let func_state = input
        .state
        .clone()
        .subvalue(state_start + StageType::BITS, func.state_len());
    // start
    let output_base = InfParOutputSys::new(config);
    let create_out_state = |s: StageType, fs| output_state.clone().concat(s.into()).concat(fs);
    if config.data_part_len <= 1 {
        assert!(temp_buffer_step >= 2);
        assert!(temp_buffer_step_pos >= 2);
    };
    // Algorithm:
    // 0: 1. Load temp_buffer data part.
    let mut output_0 = output_base.clone();
    output_0.state = create_out_state(StageType::from(1u8), func_state.clone());
    output_0.dkind = DKIND_TEMP_BUFFER.into();
    output_0.dpr = true.into();
    // 1: 2. If data_part==0: then:
    let mut output_1 = output_base.clone();
    output_1.state = create_out_state(
        int_ite(
            !(&input.dpval).bit(0),
            StageType::from(2u8),
            // go to 9.
            StageType::from(6u8),
        ),
        func_state.clone(),
    );
    // 2: 3. Move temp buffer position forward by temp_buffer_step_pos.
    let (output_2, _) = move_data_pos_stage(
        create_out_state(stage.clone(), func_state.clone()),
        create_out_state(StageType::from(3u8), func_state.clone()),
        input,
        DKIND_TEMP_BUFFER,
        DPMOVE_FORWARD,
        temp_buffer_step_pos as u64,
    );
    // 3: 4. Load mem_address data_part.
    // 5. Move forward mem_address position.
    let mut output_3 = output_base.clone();
    output_3.state = create_out_state(StageType::from(4u8), func_state.clone());
    output_3.dkind = DKIND_MEM_ADDRESS.into();
    output_3.dpr = true.into();
    output_3.dpmove = DPMOVE_FORWARD.into();
    // 4: 6. Process and store data part into current temp_buffer position.
    let (out_func_state, out_value) = func.output(func_state.clone(), input.dpval.clone());
    let mut output_4 = output_base.clone();
    output_4.state = create_out_state(StageType::from(5u8), out_func_state);
    output_4.dkind = DKIND_TEMP_BUFFER.into();
    output_4.dpw = true.into();
    output_4.dpval = out_value;
    // 5: 7. Move temp_buffer position forward by (temp_buffer_step - temp_buffer_step_pos).
    // 5: 8. Go to 1.
    let (output_5, _) = move_data_pos_stage(
        create_out_state(stage.clone(), func_state.clone()),
        create_out_state(StageType::from(0u8), func_state.clone()),
        input,
        DKIND_TEMP_BUFFER,
        DPMOVE_FORWARD,
        (temp_buffer_step - temp_buffer_step_pos) as u64,
    );
    // 9. Else (step 1)
    // 6: 10. Move temp buffer position to start.
    let (output_6, _) = data_pos_to_start_stage(
        create_out_state(stage.clone(), func_state.clone()),
        create_out_state(StageType::from(7u8), func_state.clone()),
        input,
        DKIND_TEMP_BUFFER,
    );
    // 7: 11. Move mem_address position to start.
    let (output_7, end_7) = data_pos_to_start_stage(
        create_out_state(stage.clone(), func_state.clone()),
        create_out_state(StageType::from(0u8), func_state.clone()),
        input,
        DKIND_MEM_ADDRESS,
    );
    // 12. End of algorithm.
    let end = end_7 & (&stage).equal(7u8);
    // finishing
    let output_stages = vec![
        output_0, output_1, output_2, output_3, output_4, output_5, output_6, output_7,
    ];
    finish_stage_with_table(
        output_state,
        next_state,
        input,
        output_stages,
        stage.into(),
        end,
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
    assert_eq!(output_state.bitnum(), next_state.bitnum());
    assert_ne!(temp_buffer_step, 0);
    assert_ne!(tbs_src_pos, 0);
    assert_ne!(tbs_dest_pos, 0);
    assert_ne!(tbs_src_pos, tbs_dest_pos);
    assert!(tbs_src_pos < temp_buffer_step);
    assert!(tbs_dest_pos < temp_buffer_step);
    let config = input.config();
    let dp_len = config.data_part_len as usize;
    let state_start = output_state.bitnum();
    type StageType = U4VarSys;
    extend_output_state(
        state_start,
        StageType::BITS + dp_len + 1 + func.state_len(),
        input,
    );
    let stage =
        StageType::try_from(input.state.clone().subvalue(state_start, StageType::BITS)).unwrap();
    let value = input
        .state
        .clone()
        .subvalue(state_start + StageType::BITS, dp_len);
    let dp_zero = input
        .state
        .clone()
        .bit(state_start + StageType::BITS + dp_len);
    let func_state = input
        .state
        .clone()
        .subvalue(state_start + StageType::BITS + dp_len + 1, func.state_len());
    // start
    let output_base = InfParOutputSys::new(config);
    let create_out_state = |s: StageType, v, dpz, fs| {
        output_state
            .clone()
            .concat(s.into())
            .concat(v)
            .concat(UDynVarSys::filled(1, dpz))
            .concat(fs)
    };
    let value_zero = UDynVarSys::from_n(0u8, dp_len);
    // Algorithm:
    // 0. If data_part_len == 1: Make forward temp_buffer_pos to move to proc_id end marker.
    if dp_len <= 1 {
        assert!(temp_buffer_step >= 2);
        assert!(tbs_src_pos >= 2);
        assert!(tbs_dest_pos >= 2);
    }
    // 0: 0. Read first memory end pos.
    let mut output_0 = output_base.clone();
    output_0.state = create_out_state(
        StageType::from(1u8),
        value_zero.clone(),
        dp_zero.clone(),
        func_state.clone(),
    );
    if dp_len <= 1 {
        output_0.dpmove = DPMOVE_FORWARD.into();
    }
    output_0.dkind = DKIND_TEMP_BUFFER.into();
    output_0.dpr = true.into();
    // 0_1: 1. Load temp_buffer data part.
    let mut output_0_1 = output_base.clone();
    let tidx = if dp_len <= 1 {
        output_0_1.state = create_out_state(
            if dest_proc_id_end_pos {
                StageType::from(1 + 1u8)
            } else {
                int_ite(
                    !(&input.dpval).bit(0),
                    StageType::from(1 + 1u8),
                    StageType::from(1 + 8u8),
                )
            },
            value_zero.clone(),
            if src_proc_id_end_pos {
                dp_zero.clone()
            } else {
                &dp_zero | input.dpval.bit(0)
            },
            func_state.clone(),
        );
        output_0_1.dkind = DKIND_TEMP_BUFFER.into();
        output_0_1.dpr = true.into();
        1
    } else {
        0
    };
    // 1: 2. If data_part==0: then:
    let mut output_1 = output_base.clone();
    output_1.state = create_out_state(
        if dp_len <= 1 {
            if dest_proc_id_end_pos {
                int_ite(
                    !(&input.dpval).bit(0),
                    StageType::from(tidx + 2u8),
                    // go to 8.
                    StageType::from(tidx + 8u8),
                )
            } else {
                StageType::from(tidx + 2u8)
            }
        } else {
            int_ite(
                if dest_proc_id_end_pos {
                    !(&input.dpval).bit(1)
                } else {
                    !(&input.dpval).bit(0)
                },
                StageType::from(tidx + 2u8),
                // go to 8.
                StageType::from(tidx + 8u8),
            )
        },
        value_zero.clone(),
        if dp_len <= 1 {
            if src_proc_id_end_pos {
                &dp_zero | input.dpval.bit(0)
            } else {
                dp_zero.clone()
            }
        } else {
            if src_proc_id_end_pos {
                &dp_zero | input.dpval.bit(1)
            } else {
                &dp_zero | input.dpval.bit(0)
            }
        },
        func_state.clone(),
    );
    // 2: 3. Move temp buffer position forward by tbs_src_pos.
    let (output_2, _) = move_data_pos_stage(
        create_out_state(
            stage.clone(),
            value_zero.clone(),
            dp_zero.clone(),
            func_state.clone(),
        ),
        create_out_state(
            StageType::from(tidx + 3u8),
            value_zero.clone(),
            dp_zero.clone(),
            func_state.clone(),
        ),
        input,
        DKIND_TEMP_BUFFER,
        DPMOVE_FORWARD,
        if dp_len > 1 {
            tbs_src_pos as u64
        } else {
            (tbs_src_pos - 1) as u64
        },
    );
    // 3: 4. Load temp buffer data_part from tbs_src_pos.
    let mut output_3 = output_base.clone();
    output_3.state = create_out_state(
        StageType::from(tidx + 4u8),
        value_zero.clone(),
        dp_zero.clone(),
        func_state.clone(),
    );
    output_3.dkind = DKIND_TEMP_BUFFER.into();
    output_3.dpr = true.into();
    // 4: 5. Store data part into state value
    let (out_func_state, out_value) = func.output(
        func_state.clone(),
        dynint_ite(dp_zero.clone(), value_zero.clone(), input.dpval.clone()),
    );
    let mut output_4 = output_base.clone();
    output_4.state = create_out_state(
        StageType::from(tidx + 5u8),
        out_value,
        dp_zero.clone(),
        out_func_state,
    );
    // 5: 6. Move temp_buffer position forward to tbs_dest_pos.
    let (output_5, _) = move_data_pos_stage(
        create_out_state(
            stage.clone(),
            value.clone(),
            dp_zero.clone(),
            func_state.clone(),
        ),
        create_out_state(
            StageType::from(tidx + 6u8),
            value.clone(),
            dp_zero.clone(),
            func_state.clone(),
        ),
        input,
        DKIND_TEMP_BUFFER,
        if tbs_src_pos < tbs_dest_pos {
            DPMOVE_FORWARD
        } else {
            DPMOVE_BACKWARD
        },
        if tbs_src_pos < tbs_dest_pos {
            (tbs_dest_pos - tbs_src_pos) as u64
        } else {
            (tbs_src_pos - tbs_dest_pos) as u64
        },
    );
    // 6: 7. Store value into destination temp buffer position.
    let mut output_6 = output_base.clone();
    output_6.state = create_out_state(
        StageType::from(tidx + 7u8),
        value_zero.clone(),
        dp_zero.clone(),
        func_state.clone(),
    );
    output_6.dkind = DKIND_TEMP_BUFFER.into();
    output_6.dpw = true.into();
    output_6.dpval = value.clone();
    // 7: 8. Move temp buffer position forward by (temp_buffer_step - tbs_dest_pos)
    // 8.1. Go to 1.
    let (output_7, _) = move_data_pos_stage(
        create_out_state(
            stage.clone(),
            value_zero.clone(),
            dp_zero.clone(),
            func_state.clone(),
        ),
        create_out_state(
            StageType::from(0u8),
            value_zero.clone(),
            dp_zero.clone(),
            func_state.clone(),
        ),
        input,
        DKIND_TEMP_BUFFER,
        DPMOVE_FORWARD,
        (temp_buffer_step - tbs_dest_pos) as u64,
    );
    // 9. Else (step 1)
    // 8: 10. Move temp buffer position to start.
    let (output_8, end_8) = data_pos_to_start_stage(
        create_out_state(
            stage.clone(),
            value_zero.clone(),
            dp_zero.clone(),
            func_state.clone(),
        ),
        create_out_state(
            StageType::from(0u8),
            value_zero.clone(),
            dp_zero.clone(),
            func_state.clone(),
        ),
        input,
        DKIND_TEMP_BUFFER,
    );
    // 11. End of algorithm.
    let end = end_8 & (&stage).equal(tidx + 8u8);
    // finishing
    let mut output_stages = vec![
        output_0, output_1, output_2, output_3, output_4, output_5, output_6, output_7, output_8,
    ];
    if dp_len <= 1 {
        output_stages.insert(1, output_0_1);
    }
    finish_stage_with_table(
        output_state,
        next_state,
        input,
        output_stages,
        stage.into(),
        end,
    )
}

pub fn par_process_infinite_data_stage<F: FunctionNN>(
    output_state: UDynVarSys,
    next_state: UDynVarSys,
    input: &mut InfParInputSys,
    temp_buffer_step: u32,
    src_params: &[(InfDataParam, usize)],
    dests: &[(InfDataParam, usize)],
    func: F,
) -> (InfParOutputSys, BoolVarSys) {
    assert_eq!(output_state.bitnum(), next_state.bitnum());
    assert_eq!(func.input_num(), src_params.len());
    assert_eq!(func.output_num(), dests.len());
    let config = input.config();
    let dp_len = config.data_part_len as usize;
    // src_params can be empty (no input for functions)
    assert!(!dests.is_empty());
    for (data_param, end_pos) in src_params.iter().chain(dests.iter()) {
        let good = match data_param {
            InfDataParam::TempBuffer(pos) => *pos < dp_len,
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
            .collect::<Vec<_>>();
        end_pos_words.sort();
        end_pos_words.dedup();
        end_pos_words
    };
    for (data_param, _) in src_params {
        if let InfDataParam::TempBuffer(pos) = data_param {
            // temp buffer positions shouldn't cover words with end pos markers
            assert!(end_pos_words.binary_search(pos).is_err());
        }
    }
    {
        let mut dests = dests.to_vec();
        dests.sort();
        let old_dest_len = dests.len();
        dests.dedup();
        // check whether dests have only one per different InfDataParam.
        assert_eq!(old_dest_len, dests.len());
    }
    assert!(dests
        .into_iter()
        .all(|(param, _)| *param != InfDataParam::ProcId));

    let mut total_stages = 0;
    // store all end pos limiters
    let total_state_bits = src_params.len() + dests.len();
    let mut read_state_bits = 0;
    // end_pos
    let mut use_mem_address = src_params
        .into_iter()
        .chain(dests.into_iter())
        .any(|(param, _)| *param == InfDataParam::MemAddress);
    let mut use_write_mem_address = dests
        .into_iter()
        .any(|(param, _)| *param == InfDataParam::MemAddress);
    let mut use_proc_id = src_params
        .into_iter()
        .any(|(param, _)| *param == InfDataParam::ProcId);
    let mut use_proc_id = false;
    let mut last_pos = 0;
    let mut first = true;
    for (_, end_pos) in src_params {
        let pos = *end_pos / dp_len;
        if last_pos != pos {
            total_stages += 1; // movement
            total_stages += 2; // read stage and store stage
        } else if first {
            total_stages += 2; // read stage and store stage
        }
        first = false;
        last_pos = pos;
    }
    let mut first = true;
    for (_, end_pos) in dests {
        let pos = *end_pos / dp_len;
        if last_pos != pos {
            total_stages += 1; // movement
            total_stages += 2; // read stage and store stage
        } else if first {
            total_stages += 2; // read stage and store stage
        }
        first = false;
        last_pos = pos;
    }
    // src params
    for (param, end_pos) in src_params {
        match param {
            InfDataParam::EndPos(p) => {
                let pos = *end_pos / dp_len;
                if last_pos != pos {
                    total_stages += 1; // movement stage
                }
                last_pos = pos;
                read_state_bits += 1;
            }
            InfDataParam::TempBuffer(pos) => {
                if last_pos != *pos {
                    total_stages += 1; // movement stage
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
    total_stages += 1; // process stage and store results
    let mut write_state_bits = 0;
    for (param, end_pos) in dests {
        match param {
            InfDataParam::EndPos(p) => {
                total_stages += 1; // read stage for keep values
                let pos = *end_pos / dp_len;
                if last_pos != pos {
                    total_stages += 1; // movement stage
                }
                last_pos = pos;
                write_state_bits += 1;
            }
            InfDataParam::TempBuffer(pos) => {
                if last_pos != *pos {
                    total_stages += 1; // movement stage
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
    // move to next data part
    total_stages += 1;
    // add move back stages
    let end_stage = total_stages;
    total_stages += 1 + usize::from(use_mem_address) + usize::from(use_proc_id);
    // calculate total state bits
    let total_state_bits = total_state_bits + std::cmp::max(read_state_bits, write_state_bits);

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
    let mut first = true;
    // read src_params end pos
    for (i, (_, end_pos)) in src_params.into_iter().enumerate() {
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
                    DPMOVE_FORWARD
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
            
            let mut output = output_base.clone();
            output.state = create_out_state(
                UDynVarSys::from_n(outputs.len() + 1, stage_type_len),
                state_vars.clone(),
                func_state.clone(),
            );
            output.dkind = DKIND_TEMP_BUFFER.into();
            output.dpr = true.into();
            outputs.push(output);
            // store stage
            // let mut output = output_base.clone();
            let mut new_state_vars = state_vars.clone();
            for (j, (_, end_pos)) in src_params[i..].into_iter().take_while(|(_, end_pos)| {
                let pos_2 = end_pos / dp_len;
                pos == pos_2
            }).enumerate() {
                let end_pos_val = state_vars.bit(i) | input.dpval.bit(end_pos % dp_len);
                //new_state_vars = 
            }
//             output.state = create_out_state(
//                     UDynVarSys::from_n(outputs.len() + 1, stage_type_len),
//                     UDynVarSys::from_iter(
//                         (0..state_vars.len()).map(|bit|
//     
//                     )
//                     state_vars.clone(),
//                     func_state.clone(),
//                 );
        }
        last_pos = pos;
    }

    (InfParOutputSys::new(input.config()), true.into())
}
