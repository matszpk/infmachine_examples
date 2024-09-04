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

// HINT: while moving to next position use that construction:
// state1.dpmove = dir; state2 = move_data_pos_stage(dir, pos_diff - 1),
// pos_diff - difference between two positions in temp buffer part.
// TODO: use it to any other simpler function that operates in infinite data.
// src_params and dests entry format:
// (param, end_pos):
// param_type - defined parameter position in infinite data.
// end_pos - used end position marker to limit data (limiter)
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
    // find unique words used to read
    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
    enum WordReadUsage {
        // end pos to limit other input. parameter is bit in word.
        EndPosLimit(usize),
        // end pos as input. parameter is bit in word.
        EndPosInput(usize),
        // read input from temp buffer
        TempBuffer,
    }
    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
    enum ReadOrigIndex {
        FromSrc(usize),
        FromDest(usize),
    }
    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
    struct WordReadEntry {
        pos: usize,
        usage: WordReadUsage,
        orig_index: ReadOrigIndex,
    }
    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
    enum WordWriteUsage {
        // end pos as output. parameter is bit in word.
        EndPosOutput(usize),
        // write output from temp buffer
        TempBuffer,
    }
    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
    struct WordWriteEntry {
        pos: usize,
        usage: WordWriteUsage,
        orig_index: usize,
    }
    // collect words to read from temp buffer chunk
    use ReadOrigIndex::*;
    let temp_buffer_words_to_read = {
        let mut temp_buffer_words_to_read = vec![];
        for (i, (data_param, end_pos_limit)) in src_params.into_iter().enumerate() {
            // push from end pos limiter
            temp_buffer_words_to_read.push(WordReadEntry {
                pos: end_pos_limit / dp_len,
                usage: WordReadUsage::EndPosLimit(end_pos_limit % dp_len),
                orig_index: FromSrc(i),
            });
            // push from data param
            match data_param {
                InfDataParam::EndPos(pos) => {
                    temp_buffer_words_to_read.push(WordReadEntry {
                        pos: pos / dp_len,
                        usage: WordReadUsage::EndPosInput(*pos % dp_len),
                        orig_index: FromSrc(i),
                    });
                }
                InfDataParam::TempBuffer(pos) => {
                    temp_buffer_words_to_read.push(WordReadEntry {
                        pos: *pos,
                        usage: WordReadUsage::TempBuffer,
                        orig_index: FromSrc(i),
                    });
                }
                _ => (),
            }
        }
        // push end pos limiter from destinations
        for (i, (_, end_pos_limit)) in dests.into_iter().enumerate() {
            // push from end pos limiter
            temp_buffer_words_to_read.push(WordReadEntry {
                pos: end_pos_limit / dp_len,
                usage: WordReadUsage::EndPosLimit(end_pos_limit % dp_len),
                orig_index: FromDest(i),
            });
        }
        temp_buffer_words_to_read.sort();
        temp_buffer_words_to_read.dedup();
        temp_buffer_words_to_read
    };
    // collect words to write from temp buffer chunk
    let temp_buffer_words_to_write = {
        let mut temp_buffer_words_to_write = vec![];
        for (i, (data_param, _)) in dests.into_iter().enumerate() {
            // push from data param
            match data_param {
                InfDataParam::EndPos(pos) => {
                    temp_buffer_words_to_write.push(WordWriteEntry {
                        pos: pos / dp_len,
                        usage: WordWriteUsage::EndPosOutput(*pos % dp_len),
                        orig_index: i,
                    });
                }
                InfDataParam::TempBuffer(pos) => {
                    temp_buffer_words_to_write.push(WordWriteEntry {
                        pos: *pos,
                        usage: WordWriteUsage::TempBuffer,
                        orig_index: i,
                    });
                }
                _ => (),
            }
        }
        temp_buffer_words_to_write.sort();
        temp_buffer_words_to_write
    };
    // check full filling for temp buffer pos that holds end pos to write
    // if temp buffer data part in dest with end positions is fully filled then
    // write can be done in one stage without reading data part to keep other end pos markers.
    let filled_tb_pos = {
        let mut filled_tb_pos = vec![false; temp_buffer_step as usize];
        let mut last_pos = temp_buffer_words_to_write[0].pos;
        let mut last_pos_idx = 0;
        for (i, entry) in temp_buffer_words_to_write.iter().enumerate() {
            if last_pos != entry.pos {
                if matches!(
                    temp_buffer_words_to_write[last_pos_idx].usage,
                    WordWriteUsage::EndPosOutput(_)
                ) {
                    let mut fill = vec![false; dp_len];
                    // check if all filled
                    for entry in &temp_buffer_words_to_write[last_pos_idx..i] {
                        if let WordWriteUsage::EndPosOutput(b) = entry.usage {
                            fill[b] = true;
                        }
                    }
                    // set if all bits are filled
                    filled_tb_pos[last_pos] = fill.into_iter().all(|x| x);
                }
                // new usage
                last_pos_idx = i;
            }
            last_pos = entry.pos;
        }
        // last
        if matches!(
            temp_buffer_words_to_write[last_pos_idx].usage,
            WordWriteUsage::EndPosOutput(_)
        ) {
            let mut fill = vec![false; dp_len];
            // check if all filled
            for entry in &temp_buffer_words_to_write[last_pos_idx..] {
                if let WordWriteUsage::EndPosOutput(b) = entry.usage {
                    fill[b] = true;
                }
            }
            // set if all bits are filled
            filled_tb_pos[last_pos] = fill.into_iter().all(|x| x);
        }
        filled_tb_pos
    };

    // All plan divided into 4 phases: a reading phase, a processing phase (one stage),
    // a writing stage, a moving back phase (move back data positions to start).
    // Allocation of states in order:
    // * all end pos in dests to limit writes,
    // * rest of data to read (a reading phase) or data to write (in a writing phase)
    // [DEST_END_POS,{ALL_READ_DATA|WRITE_DATA}]

    #[derive(Clone, Copy, Debug)]
    struct AllocEntry {
        param: InfDataParam,
        pos: usize,
        len: usize,
    }
    let mut read_state_bit_count = 0; // segment for rest read data.
    let mut read_pos_allocs = Vec::<AllocEntry>::new();
    let mut write_state_bit_count = 0; // this same segment as for read data.
    let mut write_pos_allocs = Vec::<AllocEntry>::new();

    // prepare plan for stages and extra states
    let use_read_mem_address_count = src_params
        .iter()
        .filter(|(dp, _)| matches!(dp, InfDataParam::MemAddress))
        .count();
    let use_write_mem_address = dests
        .iter()
        .any(|(dp, _)| matches!(dp, InfDataParam::MemAddress));
    let use_proc_id_count = src_params
        .iter()
        .filter(|(dp, _)| matches!(dp, InfDataParam::ProcId))
        .count();
    let read_mem_address_and_proc_id_stages =
        usize::from(use_read_mem_address_count != 0 && use_write_mem_address)
            + usize::from(use_proc_id_count != 0);

    // main plan of processing:
    // read mem_address data part and move postion (if not write to it)
    // read proc_id data part and move position
    // read temp buffer data parts and move positions
    // x. check if all dest end pos are set then go to end of algorithm.
    // y. filter all inputs and process all inputs.
    // x and y steps: can be done in one stage.
    // in this point it possible to fuse read/processing/write stages together
    // write mem_address data part and move position.
    // move to first temp buffer data to write
    // write temp buffer data parts and move position
    // move to start of next temp buffer data parts.
    // end of algorithm: move back to start.
    // main plain doesn't require to use variable allocator.

    // read stage scheme:
    // movement stage - stage to move this position if needed
    // read stage and movement - read temp buffer data
    //   and first movement - use read data and make first move if needed
    //   important: dpr - must be set by src end pos
    // calculation or store stage - can be fused with read stage.
    // scheme if one one move to next position:
    // [read, move]
    // [store, read, [move]] <- next read, next position
    // scheme if one two moves to next position:
    // [read, move]
    // [store, move]
    // [read,[move]].... <- next read, next position
    // scheme if one more than two moves to next position:
    // [read, move]
    // [store, move]
    // [movement stage]
    // [read, [move]] <- next read, next position

    // write stage scheme if filled endposes or other kind:
    //   if one step to next position:
    //   [write, move]
    //   [write, move] <- next position
    //   if more than one step to next position:
    //   [write, move]
    //   [movement]
    //   [write, move] <- next position:
    // write not fully filled endposes:
    //   if one step to next position:
    //   [read]
    //   [write, move] <- next position
    //   if more than one step to next position:
    //   [read]
    //   [write, move]
    //   [movement]

    // join mem_address or proc_id read with temp_buffer
    // No extra stage needed. first movement of temp buffer before reads from
    // mem_address and proc_id

    // join read and process:
    // excluded case: no temp buffer read - must be one temp buffer read.
    // first write and last read is same temp buffer position: and NO read required at write:
    // [last_read]
    // [[store],process,write,move_to_next]
    // between first write and last read only one move: and NO read required at write:
    // [last_read,move]
    // [[store],process,write,move_to_next]
    // between first write and last read more than one move: and NO read required at write:
    // [last_read,move]
    // [[store],process,movement] - need store first writes
    // [process,write,move_to_next]
    //
    // first write and last read is same temp buffer position: and read required at write:
    // [last_read] <- fuse read
    // [[store],process,write,move_to_next]
    // between first write and last read only one move: and read required at write:
    // [last_read,move]
    // [[store],process,read] - need store first writes
    // [write,move_to_next]
    // between first write and last read more than one move: and read required at write:
    // [last_read,move]
    // [[store],process,movement] - need store first writes
    // [read]
    // [write,move_to_next]
    //
    // mem_address write and last read same position as first write to temp buffer
    // and NO read required at write:
    // [last_read]
    // [[store],process,[mem_address_write],move_to_next_mem_address]
    // mem_address write and between last read and first write temp temp buffer is 1 move
    // and NO read required at write:
    // [last_read,move]
    // [[store],process,[mem_address_write],move_to_next_mem_address]
    // mem_address write and between last read and first write temp temp buffer is
    // more than one move and NO read required at write:
    // [last_read,move]
    // [[store],process,[mem_address_write],move_to_next_mem_address]
    // [movement] <- next temp buffer position

    // end_pos states - states managed in whole loop
    let (dest_end_pos_states, have_dest_end_pos_states) = {
        let first_dest_end_pos = dests[0].1;
        if dests
            .into_iter()
            .any(|(_, end_pos)| *end_pos != first_dest_end_pos)
        {
            let mut end_poses = dests
                .into_iter()
                .enumerate()
                .map(|(i, (_, end_pos))| (end_pos, i))
                .collect::<Vec<_>>();
            end_poses.sort();
            end_poses.dedup();
            (end_poses, true)
        } else {
            // no dest end pos - because one dest end control all outputs
            (vec![], false)
        }
    };
    let (src_end_pos_states, have_src_pos_states) = {
        if src_params.is_empty() ||
            // or if all src_params have same end_pos as dest endpos
            (dest_end_pos_states.is_empty() &&
                src_params.into_iter().all(|(_, end_pos)| dests[0].1 == *end_pos))
        {
            (vec![], false)
        } else {
            let mut end_poses = src_params
                .into_iter()
                .enumerate()
                .map(|(i, (_, end_pos))| (end_pos, i))
                .collect::<Vec<_>>();
            end_poses.sort();
            end_poses.dedup();
            (end_poses, true)
        }
    };

    // process reading of mem_address and proc_id.
    if use_read_mem_address_count != 0 {
        if use_proc_id_count != 0 || !temp_buffer_words_to_read.is_empty() {
            // allocate read value
            read_pos_allocs.push(AllocEntry {
                param: InfDataParam::MemAddress,
                pos: read_state_bit_count,
                len: dp_len,
            });
            read_state_bit_count += dp_len;
        }
    }
    if use_proc_id_count != 0 {
        if !temp_buffer_words_to_read.is_empty() {
            // allocate read value
            read_pos_allocs.push(AllocEntry {
                param: InfDataParam::ProcId,
                pos: read_state_bit_count,
                len: dp_len,
            });
            read_state_bit_count += dp_len;
        }
    }

    // last_pos - last read position in temp buffer
    // if in first write and if no other stage between last read and first write -
    // then get from input.dpval not from state.
    let mut last_pos_idx = 0;
    let mut total_stages = read_mem_address_and_proc_id_stages;
    let mut last_pos = 0;
    let mut first = true;
    // queue: that holds all entries with same temp buffer pos.
    // in this loop: process entry excluding last entries with different position.
    for (i, entry) in temp_buffer_words_to_read.iter().enumerate() {
        // allocate state bits
        if last_pos != entry.pos {
            // movement stage
            if (first && entry.pos != last_pos)
                || (!first &&
                    // or if requred movement to 2 next positiona requires more than one move
                    (entry.pos + 2 < last_pos
                    || entry.pos > last_pos + 2))
            {
                total_stages += 1;
            } else if !first && (entry.pos + 1 >= last_pos && entry.pos <= last_pos + 1) {
                total_stages -= 1; // store stage fusion with read stage
            }
            let mut last_usage = None;
            // in this loop: process all entries
            for entry in &temp_buffer_words_to_read[last_pos_idx..i] {
                // Important notice about ordering:
                // Next majority after usage is enum's variant (FromDest and FromSrc)
                // thus, ordering of temp_buffer_words_to_read is correct.
                // (FromDest, FromSrc).
                let cur_usage = (entry.usage, matches!(entry.orig_index, FromDest(_)));
                if Some(cur_usage) != last_usage {
                    // process first entry with different usage or source of original index.
                    if let FromSrc(p) = entry.orig_index {
                        match entry.usage {
                            WordReadUsage::EndPosInput(b) => {
                                // the allocate in read segment
                                read_pos_allocs.push(AllocEntry {
                                    param: InfDataParam::EndPos(dp_len * p + b),
                                    pos: read_state_bit_count,
                                    len: 1,
                                });
                                read_state_bit_count += 1;
                            }
                            WordReadUsage::TempBuffer => {
                                read_pos_allocs.push(AllocEntry {
                                    param: InfDataParam::TempBuffer(p),
                                    pos: read_state_bit_count,
                                    len: dp_len,
                                });
                                read_state_bit_count += dp_len;
                            }
                            _ => (),
                        }
                    }
                    last_usage = Some(cur_usage);
                }
            }
            // reading
            if first || entry.pos != last_pos {
                total_stages += 2; // include read stage and store stage.
            }
            last_pos_idx = i;
        }
        // rest of iteration
        last_pos = entry.pos;
        first = false;
    }
    // determine next temp buffer position
    let next_phase_position = if !temp_buffer_words_to_write.is_empty() {
        temp_buffer_words_to_write[0].pos
    } else {
        temp_buffer_step as usize
    };
    let fuse_read_with_write_temp_buffer_end_pos_read = {
        if let Some(write) = temp_buffer_words_to_write.get(0) {
            last_pos == write.pos && matches!(write.usage, WordWriteUsage::EndPosOutput(_))
        } else {
            false
        }
    };
    // determine join with process stage and first write stage
    let join_with_first_write_stage = use_write_mem_address
        // if only one move needed to first write position
        || ((next_phase_position + 1 >= last_pos
            && next_phase_position <= last_pos + 1) && (
            if let Some(write) = temp_buffer_words_to_write.get(0) {
                // if whole temp buffer data part
                (matches!(write.usage, WordWriteUsage::TempBuffer)
                    // or filled by all end pos outputs or fused with read between
                    // last_read and first_write.
                    || (fuse_read_with_write_temp_buffer_end_pos_read || filled_tb_pos[write.pos]))
            } else {
                false
            }));
    //
    total_stages += 2; // include read stage and process stage.

    // allocate writes excluding first write
    if !temp_buffer_words_to_write.is_empty() {
        let write_first_pos = temp_buffer_words_to_write[0].pos;
        // skip_while - entry.pos == skip if write_first_pos - skip first write.
        // skip if possible join first write stage with process stage - then skip
        //     first writes can be ommited while storing in states (it read directly).
        // otherwise all writes should be stored in states because all will be in next stages.
        // use: join_with_first_write_stage && e.pos == write_first_pos to do it.
        for entry in temp_buffer_words_to_write
            .iter()
            .skip_while(|e| join_with_first_write_stage && e.pos == write_first_pos)
        {
            // entries in temp_buffer_words_to_write are unique (unique position with usage).
            // just process single writes
            match entry.usage {
                WordWriteUsage::EndPosOutput(b) => {
                    write_pos_allocs.push(AllocEntry {
                        param: InfDataParam::EndPos(entry.pos * dp_len + b),
                        pos: write_state_bit_count,
                        len: 1,
                    });
                    write_state_bit_count += 1;
                }
                WordWriteUsage::TempBuffer => {
                    write_pos_allocs.push(AllocEntry {
                        param: InfDataParam::TempBuffer(entry.pos),
                        pos: write_state_bit_count,
                        len: dp_len,
                    });
                    write_state_bit_count += dp_len;
                }
            }
        }
    }

    // now first memory address write. If done then should be fused with
    // store and process stage and include total_stages.

    // join last process and first write to one stage if:
    // * first write to memory_address
    // * first write to temp buffer is to position at most 1 move forward or backward
    //   * write is filled end pos temp buffer position or temp_buffer write
    if join_with_first_write_stage {
        total_stages -= 1; // join write with process stage
    }

    // prepare stages for write words
    let mut first = true;
    let mut write_last_pos = None;
    // in this loop: all entries
    for entry in &temp_buffer_words_to_write {
        if Some(entry.pos) != write_last_pos {
            // at first write include first move at last read
            if entry.pos + 1 < last_pos || entry.pos > last_pos + 1 {
                total_stages += 1;
            }
            if !filled_tb_pos[entry.pos] {
                if !first && !fuse_read_with_write_temp_buffer_end_pos_read {
                    // exclude special case when last read position == first write position
                    // and no memory write - then fuse read with last read from read phase.
                    total_stages += 1; // add if not filled, and read stage needed
                }
            }
            total_stages += 1;
            last_pos = entry.pos;
            write_last_pos = Some(entry.pos);
            first = false;
        }
    }
    // Now. Add to total_stages stage to move to next data chunk at start.
    // (temp_buffer_step - last_pos)
    // at first write include first move at last read
    if (temp_buffer_step as usize) + 1 >= last_pos && (temp_buffer_step as usize) <= last_pos + 1 {
        total_stages += 1;
    }
    let end_stage = total_stages;

    // stages to move backwards. if any DataParam is MemAddress or ProcId then add 1.
    total_stages += 1 + read_mem_address_and_proc_id_stages;
    let total_stages = total_stages; // as not mutable (read-only)

    // fix allocs
    for AllocEntry { pos: t, .. } in &mut read_pos_allocs {
        *t += src_end_pos_states.len() + dest_end_pos_states.len();
    }
    for AllocEntry { pos: t, .. } in &mut write_pos_allocs {
        *t += src_end_pos_states.len() + dest_end_pos_states.len();
    }
    let state_bit_num = src_end_pos_states.len()
        + dest_end_pos_states.len()
        + std::cmp::max(read_state_bit_count, write_state_bit_count);
    // sort allocs
    read_pos_allocs.sort_by_key(|x| x.param);
    write_pos_allocs.sort_by_key(|x| x.param);

    //
    // MAIN PROCESS:
    // circuit generation: stages generation.
    //
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct EndPosStateEntry {
        index: Option<usize>,
        end_pos: usize,
        from_dest: bool,
        val: BoolVarSys,
    };
    let state_start = output_state.bitnum();
    let stage_type_len = calc_log_bits(total_stages);
    extend_output_state(
        state_start,
        stage_type_len + state_bit_num + func.state_len(),
        input,
    );
    let default_state_vars = input
        .state
        .clone()
        .subvalue(state_start + stage_type_len, state_bit_num);
    let apply_to_state_vars =
        |allocs: &[AllocEntry], ov: &UDynVarSys, vs: &[(InfDataParam, UDynVarSys)]| {
            // get value, pos and lengths tuple
            let mut val_and_pos = vs
                .into_iter()
                .map(|(param, v)| {
                    let p = allocs.binary_search_by_key(param, |x| x.param).unwrap();
                    (read_pos_allocs[p].pos, read_pos_allocs[p].len, v.clone())
                })
                .collect::<Vec<_>>();
            // sort by position
            val_and_pos.sort_by_key(|(p, _, _)| *p);
            let mut start = 0;
            let mut bitvec = vec![];
            // construct bit vector
            for (pos, len, v) in val_and_pos {
                bitvec.extend((start..pos).map(|i| ov.bit(i)));
                bitvec.extend((0..len).map(|i| v.bit(i)));
                start = pos + len;
            }
            bitvec.extend((start..ov.len()).map(|i| ov.bit(i)));
            // to dynintvar
            UDynVarSys::from_iter(bitvec)
        };
    let new_end_pos_states = |last_pos_idx, i, input: &InfParInputSys| {
        let mut end_poses = temp_buffer_words_to_read[last_pos_idx..i]
            .iter()
            .filter_map(|e| match e.usage {
                WordReadUsage::EndPosLimit(b) => {
                    let end_pos = last_pos * dp_len + b;
                    let (end_pos_states, from_dest) = match e.orig_index {
                        FromSrc(_) => (&src_end_pos_states, false),
                        FromDest(_) => (&dest_end_pos_states, true),
                    };
                    // find end_pos in state - p is index in states
                    if let Ok(p) = end_pos_states.binary_search_by_key(&end_pos, |(x, _)| **x) {
                        Some(EndPosStateEntry {
                            index: Some(p), // first element in tuple is index of bit in states
                            val: default_state_vars.bit(p) | input.dpval.bit(b),
                            end_pos,
                            from_dest,
                        })
                    } else {
                        // if special case - no end pos state. index in states is undefined
                        Some(EndPosStateEntry {
                            index: None,
                            val: input.dpval.bit(b),
                            end_pos,
                            from_dest,
                        })
                    }
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        end_poses.sort_by_key(|x| x.index);
        end_poses
    };
    let get_updates_for_reads = |prev_reads: &[InfDataParam], input: &InfParInputSys| {
        prev_reads
            .iter()
            .map(|param| {
                (
                    *param,
                    match param {
                        InfDataParam::EndPos(p) => {
                            UDynVarSys::filled(1, input.dpval.bit(p % dp_len))
                        }
                        InfDataParam::MemAddress
                        | InfDataParam::ProcId
                        | InfDataParam::TempBuffer(_) => input.dpval.clone(),
                    },
                )
            })
            .collect::<Vec<_>>()
    };
    let update_end_pos_states = |ov: &UDynVarSys, new_vals: &[EndPosStateEntry]| {
        let mut start = 0;
        let mut bitvec = vec![];
        // construct bit vector
        for EndPosStateEntry {
            index: pos, val: v, ..
        } in new_vals
        {
            if let Some(pos) = pos {
                // pos - index in bit in states
                bitvec.extend((start..*pos).map(|i| ov.bit(i)));
                bitvec.push(v.clone());
                start = *pos + 1;
            }
        }
        bitvec.extend((start..ov.len()).map(|i| ov.bit(i)));
        // to dynintvar
        UDynVarSys::from_iter(bitvec)
    };
    let func_state = input.state.clone().subvalue(
        state_start + stage_type_len + state_bit_num,
        func.state_len(),
    );
    // create_out_state: params: s - stage, sv - state_vars, fs - function state
    let create_out_state = |s, sv, fs| output_state.clone().concat(s).concat(sv).concat(fs);
    let output_base = InfParOutputSys::new(config);
    let mut outputs = vec![];
    // previous read to store
    let mut prev_reads = Vec::<InfDataParam>::new();
    let mut prev_end_pos_states = vec![];

    if temp_buffer_words_to_read.is_empty() && temp_buffer_words_to_read[0].pos != 0 {
        // make first movement of temp buffer position
        outputs.push(
            move_data_pos_stage(
                create_out_state(
                    UDynVarSys::from_n(outputs.len(), stage_type_len),
                    default_state_vars.clone(),
                    func_state.clone(),
                ),
                create_out_state(
                    UDynVarSys::from_n(outputs.len() + 1, stage_type_len),
                    default_state_vars.clone(),
                    func_state.clone(),
                ),
                input,
                DKIND_TEMP_BUFFER,
                DPMOVE_FORWARD,
                temp_buffer_words_to_read[0].pos as u64,
            )
            .0,
        );
    }
    if use_read_mem_address_count != 0 {
        let mut output = output_base.clone();
        output.state = create_out_state(
            UDynVarSys::from_n(outputs.len(), stage_type_len),
            default_state_vars.clone(),
            func_state.clone(),
        );
        output.dkind = DKIND_MEM_ADDRESS.into();
        if !use_write_mem_address {
            output.dpmove = DPMOVE_FORWARD.into();
        }
        output.dpr = true.into();
        outputs.push(output);
        // set previous reads
        prev_reads = vec![InfDataParam::MemAddress];
    }
    if use_proc_id_count != 0 {
        // update state vars for previous read
        let state_vars = if use_read_mem_address_count != 0 {
            apply_to_state_vars(
                &read_pos_allocs,
                &default_state_vars,
                &[(InfDataParam::MemAddress, input.dpval.clone())],
            )
        } else {
            default_state_vars.clone()
        };
        prev_reads.clear(); // prev_reads already used
        let mut output = output_base.clone();
        output.state = create_out_state(
            UDynVarSys::from_n(outputs.len(), stage_type_len),
            state_vars,
            func_state.clone(),
        );
        output.dkind = DKIND_PROC_ID.into();
        output.dpmove = DPMOVE_FORWARD.into();
        output.dpr = true.into();
        outputs.push(output);
        // set previous reads
        prev_reads = vec![InfDataParam::ProcId];
    }

    // main loop of read phase
    let mut cur_pos = 0;
    let mut last_pos = 0;
    let mut last_pos_idx = 0;
    let mut first = true;
    let mut stop_processed = false;
    // in this loop: process entry excluding last entries with different position.
    for (i, entry) in temp_buffer_words_to_read.iter().enumerate() {
        // rest of iteration
        if entry.pos != last_pos {
            if cur_pos != entry.pos {
                // make movement of temp buffer position before read
                outputs.push(
                    move_data_pos_stage(
                        create_out_state(
                            UDynVarSys::from_n(outputs.len(), stage_type_len),
                            default_state_vars.clone(),
                            func_state.clone(),
                        ),
                        create_out_state(
                            UDynVarSys::from_n(outputs.len() + 1, stage_type_len),
                            default_state_vars.clone(),
                            func_state.clone(),
                        ),
                        input,
                        DKIND_TEMP_BUFFER,
                        if cur_pos < entry.pos {
                            DPMOVE_FORWARD
                        } else {
                            DPMOVE_BACKWARD
                        },
                        if cur_pos < entry.pos {
                            entry.pos - cur_pos
                        } else {
                            cur_pos - entry.pos
                        } as u64,
                    )
                    .0,
                );
                cur_pos = entry.pos;
            }
            // determine next position
            let next_pos = {
                let count = temp_buffer_words_to_read[i..]
                    .iter()
                    .take_while(|e| e.pos == entry.pos)
                    .count();
                if i + count < temp_buffer_words_to_read.len() {
                    temp_buffer_words_to_read[i + count].pos
                } else {
                    next_phase_position
                }
            };
            for read_stage in 0..2 {
                if read_stage == 1 && cur_pos == next_pos {
                    // if next stage - to store and no more movement to next position.
                    // then skip this stage
                    // create end_pos_states if for next reading
                    prev_end_pos_states = new_end_pos_states(last_pos_idx, i, input);
                    continue;
                }
                // read stage and store previous reads
                let mut output = output_base.clone();
                let mut new_state = default_state_vars.clone();
                // update stage
                new_state = apply_to_state_vars(
                    &read_pos_allocs,
                    &new_state,
                    &get_updates_for_reads(&prev_reads, input),
                );
                prev_reads.clear();
                // update new end pos states
                new_state = update_end_pos_states(&new_state, &prev_end_pos_states);
                // special case of end pos - only one unique end pos:
                let dest_end_pos_pos = prev_end_pos_states.iter().position(
                    |EndPosStateEntry {
                         end_pos, from_dest, ..
                     }| *end_pos == dests[0].1 && *from_dest,
                );
                let next_stage = if !prev_end_pos_states.is_empty()
                    && !have_dest_end_pos_states
                    && dest_end_pos_pos.is_some()
                    && !stop_processed
                {
                    // check and if end then go to end_stage
                    stop_processed = true;
                    dynint_ite(
                        prev_end_pos_states[dest_end_pos_pos.unwrap()].val.clone(),
                        UDynVarSys::from_n(end_stage, stage_type_len),
                        UDynVarSys::from_n(outputs.len(), stage_type_len),
                    )
                } else {
                    UDynVarSys::from_n(outputs.len(), stage_type_len)
                };
                prev_end_pos_states.clear();
                // output setup
                output.state = create_out_state(next_stage, new_state, func_state.clone());
                output.dkind = DKIND_TEMP_BUFFER.into();
                if cur_pos != next_pos {
                    // make move to next position
                    output.dpmove = if cur_pos < next_pos {
                        cur_pos += 1;
                        DPMOVE_FORWARD.into()
                    } else {
                        cur_pos -= 1;
                        DPMOVE_BACKWARD.into()
                    };
                }
                if read_stage == 0 {
                    // if store_stage then no reading
                    output.dpr = true.into();
                }
                outputs.push(output);
                if read_stage == 1 {
                    // create end_pos_states if next stage
                    prev_end_pos_states = new_end_pos_states(last_pos_idx, i, input);
                }
            }
            last_pos_idx = i;
        }
        last_pos = entry.pos;
        first = false;
    }
    // processing state
    //
    let mut output = output_base.clone();
    // update new end pos states
    let mut new_state = update_end_pos_states(&default_state_vars, &prev_end_pos_states);
    // special case of end pos - only one unique end pos:
    let dest_end_pos_pos = prev_end_pos_states.iter().position(
        |EndPosStateEntry {
             end_pos, from_dest, ..
         }| *end_pos == dests[0].1 && *from_dest,
    );
    let next_stage = if !prev_end_pos_states.is_empty()
        && !have_dest_end_pos_states
        && dest_end_pos_pos.is_some()
        && !stop_processed
    {
        // check and if end then go to end_stage
        stop_processed = true;
        dynint_ite(
            prev_end_pos_states[dest_end_pos_pos.unwrap()].val.clone(),
            UDynVarSys::from_n(end_stage, stage_type_len),
            UDynVarSys::from_n(outputs.len(), stage_type_len),
        )
    } else {
        UDynVarSys::from_n(outputs.len(), stage_type_len)
    };
    // prepare inputs for processing function
    let func_inputs = src_params
        .into_iter()
        .map(|(param, end_pos)| {
            let val = if let Ok(p) = read_pos_allocs.binary_search_by_key(&param, |x| &x.param) {
                // if stored
                let pos = read_pos_allocs[p].pos;
                let len = read_pos_allocs[p].len;
                UDynVarSys::from_iter((0..len).map(|i| default_state_vars.bit(pos + i)))
            } else {
                // if in input
                match param {
                    InfDataParam::MemAddress
                    | InfDataParam::ProcId
                    | InfDataParam::TempBuffer(_) => input.dpval.clone(),
                    InfDataParam::EndPos(b) => UDynVarSys::filled(1, input.dpval.bit(b % dp_len)),
                }
            };
            // get source end pos
            let src_end_limiter = if let Some(p) = prev_end_pos_states
                .iter()
                .find(|e| !e.from_dest && e.end_pos == *end_pos)
            {
                // if from last read
                p.val.clone()
            } else {
                // if stored in state vars
                let p = src_end_pos_states
                    .binary_search_by_key(end_pos, |(x_end_pos, _)| **x_end_pos)
                    .unwrap();
                default_state_vars.bit(p)
            };
            // filter it
            dynint_ite(
                !src_end_limiter,
                val.clone(),
                UDynVarSys::from_n(0u8, val.bitnum()),
            )
        })
        .collect::<Vec<_>>();
    // clear before deterime function input to filter
    prev_end_pos_states.clear();
    // now call process function
    let (next_func_state, out_values) = func.output(func_state.clone(), &func_inputs);
    // update output states
    let mut out_to_update = vec![];
    let mut can_move_temp_buffer = true;
    let mut if_get_end_pos_data_part = false; // if not
    for ((param, _), outval) in dests.into_iter().zip(out_values) {
        if write_pos_allocs
            .binary_search_by_key(param, |aentry| aentry.param)
            .is_ok()
        {
            // write to states
            out_to_update.push((*param, outval.clone()));
        } else {
            // write to output
            match param {
                InfDataParam::MemAddress => {
                    can_move_temp_buffer = false;
                    output.dkind = DKIND_MEM_ADDRESS.into();
                    output.dpmove = DPMOVE_FORWARD.into();
                    output.dpval = outval.clone();
                }
                InfDataParam::TempBuffer(_) => {
                    output.dpval = outval.clone();
                }
                InfDataParam::EndPos(p) => {
                    let end_pos_bit = p % dp_len;
                    if !if_get_end_pos_data_part {
                        // use input data part (last read)
                        output.dpval = UDynVarSys::from_iter((0..dp_len).map(|i| {
                            if end_pos_bit != i {
                                input.dpval.bit(i)
                            } else {
                                outval.bit(0)
                            }
                        }));
                        if_get_end_pos_data_part = true;
                    } else {
                        // use current output datapart
                        output.dpval = UDynVarSys::from_iter((0..dp_len).map(|i| {
                            if end_pos_bit != i {
                                input.dpval.bit(i)
                            } else {
                                output.dpval.bit(i)
                            }
                        }));
                    }
                }
                _ => (),
            }
        }
    }
    // update state vars by new writes values.
    new_state = apply_to_state_vars(&write_pos_allocs, &new_state, &out_to_update);
    // output setup
    output.state = create_out_state(next_stage, new_state, next_func_state);
    if can_move_temp_buffer {
        output.dkind = DKIND_TEMP_BUFFER.into();
        if cur_pos != next_phase_position {
            // make move to next position
            output.dpmove = if cur_pos < next_phase_position {
                cur_pos += 1;
                DPMOVE_FORWARD.into()
            } else {
                cur_pos -= 1;
                DPMOVE_BACKWARD.into()
            };
        }
    }

    (InfParOutputSys::new(input.config()), true.into())
}
