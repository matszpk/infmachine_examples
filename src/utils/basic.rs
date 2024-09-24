use std::collections::HashMap;

use super::*;

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

//

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

pub fn finish_machine_with_table(
    mut mobj: InfParMachineObjectSys,
    input: &InfParInputSys,
    mut output_stages: Vec<InfParOutputSys>,
    stage: UDynVarSys,
) -> InfParMachineDataSys {
    InfParOutputSys::fix_state_len(&mut output_stages);
    let output_stages = output_stages
        .into_iter()
        .map(|v| {
            let state_int = v.to_dynintvar();
            state_int.subvalue(0, state_int.bitnum())
        })
        .collect::<Vec<_>>();
    let last = UDynVarSys::from_n(0u8, output_stages[0].bitnum());
    // Use output state outside joining outputs to reduce gates. It is possible because
    // first outputs are state outputs.
    let final_state = dynint_table_partial(stage, output_stages, last);
    mobj.in_state = Some(input.state.clone());
    mobj.from_dynintvar(final_state);
    mobj.to_machine()
}

// install external outputs in stage.
// ext_out_start - start (in bits) in output state
// input_state - input state
// ext_outs - ext output to replace
// ext_out_set - true if ext output should be set
pub fn install_external_outputs(
    output: InfParOutputSys,
    ext_out_start: usize,
    input_state: &UDynVarSys,
    ext_outs: UDynVarSys,
    ext_out_set: BoolVarSys,
) -> InfParOutputSys {
    let mut output = output.clone();
    let ext_out_len = ext_outs.bitnum();
    output.state = UDynVarSys::from_iter((0..output.state.bitnum()).map(|i| {
        if ext_out_start <= i && i < ext_out_start + ext_out_len {
            // install this ext_outputs bit to state
            bool_ite(
                ext_out_set.clone(),
                ext_outs.bit(i - ext_out_start),
                input_state.bit(i),
            )
        } else {
            output.state.bit(i)
        }
    }));
    output
}

// LabelHandler - object that handle jumps between stages

pub fn gen_label(place: &str, id: usize, label: &str) -> String {
    format!("{}_{}:{}", place, id, label)
}

pub struct LabelHandler {
    second_pass: bool,
    map: HashMap<String, Option<u64>>,
}

impl LabelHandler {
    pub fn new() -> Self {
        Self {
            second_pass: false,
            map: HashMap::new(),
        }
    }
    pub fn next_pass(&mut self) {
        self.second_pass = true;
    }
    pub fn add_label(&mut self, name: &str) {
        if !self.second_pass {
            self.map.insert(name.to_string(), None);
        }
    }
    pub fn set_label(&mut self, name: &str, pos: u64) {
        if !self.second_pass {
            if let Some(v) = self.map.get_mut(name) {
                if v.is_none() {
                    *v = Some(pos);
                } else {
                    panic!("Label {} already set", name);
                }
            } else {
                panic!("Label {} doesn't exist", name);
            }
        }
    }
    pub fn label(&mut self, name: &str) -> u64 {
        if self.second_pass {
            self.map[name].unwrap()
        } else {
            0
        }
    }
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
