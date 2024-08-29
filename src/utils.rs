use gategen::boolvar::*;
use gategen::dynintvar::*;
use gategen::intvar::*;
use infmachine_gen::*;

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
    assert_eq!(output_state.bitnum(), next_state.bitnum());
    assert_ne!(temp_buffer_step, 0);
    assert_ne!(temp_buffer_step_pos, 0);
    assert!(temp_buffer_step_pos < temp_buffer_step);
    let config = input.config();
    let dp_len = config.data_part_len as usize;
    let state_start = output_state.bitnum();
    type StageType = U4VarSys;
    extend_output_state(state_start, StageType::BITS, input);
    let stage =
        StageType::try_from(input.state.clone().subvalue(state_start, StageType::BITS)).unwrap();
    // start
    let output_base = InfParOutputSys::new(config);
    let create_out_state = |s: StageType| output_state.clone().concat(s.into());
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
        output_tshift.state = create_out_state(StageType::from(1u8));
        output_tshift.dpmove = U2VarSys::from(DPMOVE_FORWARD);
        output_tshift.dkind = DKIND_TEMP_BUFFER.into();
    }
    // 0: 1. Load temp_buffer data part.
    let mut output_0 = output_base.clone();
    output_0.state = create_out_state(StageType::from(tidx + 1u8));
    output_0.dkind = DKIND_TEMP_BUFFER.into();
    output_0.dpr = true.into();
    // 1: 2. If data_part==0: then:
    let mut output_1 = output_base.clone();
    let no_end_of_proc_id = if dp_len >= 2 {
        !(&input.dpval).bit(1)
    } else {
        !(&input.dpval).bit(0)
    };
    output_1.state = create_out_state(int_ite(
        no_end_of_proc_id,
        StageType::from(tidx + 2u8),
        // go to 9.
        StageType::from(tidx + 6u8),
    ));
    // 2: 3. Move temp buffer position forward by temp_buffer_step_pos.
    let (output_2, _) = move_data_pos_stage(
        create_out_state(stage.clone()),
        create_out_state(StageType::from(tidx + 3u8)),
        input,
        DKIND_TEMP_BUFFER,
        DPMOVE_FORWARD,
        temp_buffer_step_pos as u64,
    );
    // 3: 4. Load proc_id data_part.
    // 5. Move forward proc id position.
    let mut output_3 = output_base.clone();
    output_3.state = create_out_state(StageType::from(tidx + 4u8));
    output_3.dkind = DKIND_PROC_ID.into();
    output_3.dpr = true.into();
    output_3.dpmove = DPMOVE_FORWARD.into();
    // 4: 6. Store data part into current temp buffer position.
    let mut output_4 = output_base.clone();
    output_4.state = create_out_state(StageType::from(tidx + 5u8));
    output_4.dkind = DKIND_TEMP_BUFFER.into();
    output_4.dpw = true.into();
    output_4.dpval = input.dpval.clone();
    // 5: 7. Move temp_buffer position forward by (temp_buffer_step - temp_buffer_step_pos).
    // 5: 8. Go to 1.
    let (output_5, _) = move_data_pos_stage(
        create_out_state(stage.clone()),
        create_out_state(StageType::from(tidx)),
        input,
        DKIND_TEMP_BUFFER,
        DPMOVE_FORWARD,
        (temp_buffer_step - temp_buffer_step_pos) as u64,
    );
    // 9. Else (step 1)
    // 6: 10. Move temp buffer position to start.
    let (output_6, _) = data_pos_to_start_stage(
        create_out_state(stage.clone()),
        create_out_state(StageType::from(tidx + 7u8)),
        input,
        DKIND_TEMP_BUFFER,
    );
    // 7: 11. Move proc id position to start.
    let (output_7, end_7) = data_pos_to_start_stage(
        create_out_state(stage.clone()),
        create_out_state(StageType::from(0u8)),
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

// par_copy_proc_id_to_mem_address_stage - copy proc_id to mem_address.
// Include mem_address_pos_end.
pub fn par_copy_proc_id_to_mem_address_stage(
    output_state: UDynVarSys,
    next_state: UDynVarSys,
    input: &mut InfParInputSys,
    temp_buffer_step: u32,
) -> (InfParOutputSys, BoolVarSys) {
    assert_eq!(output_state.bitnum(), next_state.bitnum());
    assert_ne!(temp_buffer_step, 0);
    let config = input.config();
    let dp_len = config.data_part_len as usize;
    let state_start = output_state.bitnum();
    type StageType = U3VarSys;
    extend_output_state(state_start, StageType::BITS, input);
    let stage =
        StageType::try_from(input.state.clone().subvalue(state_start, StageType::BITS)).unwrap();
    // start
    let output_base = InfParOutputSys::new(config);
    let create_out_state = |s: StageType| output_state.clone().concat(s.into());
    if config.data_part_len <= 1 {
        assert!(temp_buffer_step >= 2);
    };
    // Algorithm:
    // 0: 1. Load temp_buffer data part.
    let mut output_0 = output_base.clone();
    output_0.state = create_out_state(StageType::from(1u8));
    output_0.dkind = DKIND_TEMP_BUFFER.into();
    output_0.dpr = true.into();
    // 1: 2. If data_part==0: then:
    let mut output_1 = output_base.clone();
    output_1.state = create_out_state(int_ite(
        !(&input.dpval).bit(0),
        StageType::from(2u8),
        // go to 9.
        StageType::from(5u8),
    ));
    // 2: 3. Load proc_id data_part.
    // 2: 4. Move forward proc id position.
    let mut output_2 = output_base.clone();
    output_2.state = create_out_state(StageType::from(3u8));
    output_2.dkind = DKIND_PROC_ID.into();
    output_2.dpr = true.into();
    output_2.dpmove = DPMOVE_FORWARD.into();
    // 3: 4. Store data part into current temp buffer position.
    // 3: 5. Move mem_address position forward.
    let mut output_3 = output_base.clone();
    output_3.state = create_out_state(StageType::from(4u8));
    output_3.dkind = DKIND_MEM_ADDRESS.into();
    output_3.dpw = true.into();
    output_3.dpval = input.dpval.clone();
    output_3.dpmove = DPMOVE_FORWARD.into();
    // 4: 7. Move temp_buffer position forward by temp_buffer_step.
    // 4: 8. Go to 1.
    let (output_4, _) = move_data_pos_stage(
        create_out_state(stage.clone()),
        create_out_state(StageType::from(0u8)),
        input,
        DKIND_TEMP_BUFFER,
        DPMOVE_FORWARD,
        temp_buffer_step as u64,
    );
    // 9. Else (step 1)
    // 5: 10. Move mem address position to start.
    let (output_5, _) = data_pos_to_start_stage(
        create_out_state(stage.clone()),
        create_out_state(StageType::from(6u8)),
        input,
        DKIND_MEM_ADDRESS,
    );
    // 6: 11. Move temp buffer position to start.
    let (output_6, _) = data_pos_to_start_stage(
        create_out_state(stage.clone()),
        create_out_state(StageType::from(7u8)),
        input,
        DKIND_TEMP_BUFFER,
    );
    // 7: 12. Move proc id position to start.
    let (output_7, end_7) = data_pos_to_start_stage(
        create_out_state(stage.clone()),
        create_out_state(StageType::from(0u8)),
        input,
        DKIND_PROC_ID,
    );
    // 13. End of algorithm.
    let end = end_7 & (&stage).equal(7u8);
    // finishing
    let mut output_stages = vec![
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

// par_copy_temp_buffer_to_mem_address_stage - copy temp_buffer to mem_address
// Include mem_address_pos_end.
pub fn par_copy_temp_buffer_to_mem_address_stage(
    output_state: UDynVarSys,
    next_state: UDynVarSys,
    input: &mut InfParInputSys,
    temp_buffer_step: u32,
    temp_buffer_step_pos: u32,
) -> (InfParOutputSys, BoolVarSys) {
    assert_eq!(output_state.bitnum(), next_state.bitnum());
    assert_ne!(temp_buffer_step, 0);
    assert_ne!(temp_buffer_step_pos, 0);
    assert!(temp_buffer_step_pos < temp_buffer_step);
    let config = input.config();
    let dp_len = config.data_part_len as usize;
    let state_start = output_state.bitnum();
    type StageType = U3VarSys;
    extend_output_state(state_start, StageType::BITS, input);
    let stage =
        StageType::try_from(input.state.clone().subvalue(state_start, StageType::BITS)).unwrap();
    // start
    let output_base = InfParOutputSys::new(config);
    let create_out_state = |s: StageType| output_state.clone().concat(s.into());
    if config.data_part_len <= 1 {
        assert!(temp_buffer_step >= 2);
        assert!(temp_buffer_step_pos >= 2);
    };
    // Algorithm:
    // 0: 1. Load temp_buffer data part.
    let mut output_0 = output_base.clone();
    output_0.state = create_out_state(StageType::from(1u8));
    output_0.dkind = DKIND_TEMP_BUFFER.into();
    output_0.dpr = true.into();
    // 1: 2. If data_part==0: then:
    let mut output_1 = output_base.clone();
    output_1.state = create_out_state(int_ite(
        !(&input.dpval).bit(0),
        StageType::from(2u8),
        // go to 9.
        StageType::from(6u8),
    ));
    // 2: 3. Move temp buffer position forward by temp_buffer_step_pos.
    let (output_2, _) = move_data_pos_stage(
        create_out_state(stage.clone()),
        create_out_state(StageType::from(3u8)),
        input,
        DKIND_TEMP_BUFFER,
        DPMOVE_FORWARD,
        temp_buffer_step_pos as u64,
    );
    // 3: 4. Load temp_buffer data_part.
    let mut output_3 = output_base.clone();
    output_3.state = create_out_state(StageType::from(4u8));
    output_3.dkind = DKIND_TEMP_BUFFER.into();
    output_3.dpr = true.into();
    // 4: 5. Store data part into current mem_address position.
    // 6. Move forward proc id position.
    let mut output_4 = output_base.clone();
    output_4.state = create_out_state(StageType::from(5u8));
    output_4.dkind = DKIND_MEM_ADDRESS.into();
    output_4.dpw = true.into();
    output_4.dpval = input.dpval.clone();
    output_3.dpmove = DPMOVE_FORWARD.into();
    // 5: 7. Move temp_buffer position forward by (temp_buffer_step - temp_buffer_step_pos).
    // 5: 8. Go to 1.
    let (output_5, _) = move_data_pos_stage(
        create_out_state(stage.clone()),
        create_out_state(StageType::from(0u8)),
        input,
        DKIND_TEMP_BUFFER,
        DPMOVE_FORWARD,
        (temp_buffer_step - temp_buffer_step_pos) as u64,
    );
    // 9. Else (step 1)
    // 6: 10. Move temp buffer position to start.
    let (output_6, _) = data_pos_to_start_stage(
        create_out_state(stage.clone()),
        create_out_state(StageType::from(7u8)),
        input,
        DKIND_TEMP_BUFFER,
    );
    // 7: 11. Move proc id position to start.
    let (output_7, end_7) = data_pos_to_start_stage(
        create_out_state(stage.clone()),
        create_out_state(StageType::from(0u8)),
        input,
        DKIND_MEM_ADDRESS,
    );
    // 12. End of algorithm.
    let end = end_7 & (&stage).equal(7u8);
    // finishing
    let mut output_stages = vec![
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
