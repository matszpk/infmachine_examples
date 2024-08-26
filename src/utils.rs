use gategen::boolvar::*;
use gategen::dynintvar::*;
use gategen::intvar::*;
use infmachine_config::*;
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

pub fn extend_output_state(
    state_start: usize,
    extra_bits: usize,
    input: &InfParInputSys,
) -> UDynVarSys {
    assert!(state_start <= input.state.bitnum());
    if state_start + extra_bits > input.state.bitnum() {
        input.state.clone().concat(UDynVarSys::var(
            state_start + extra_bits - input.state.bitnum(),
        ))
    } else {
        input.state.clone()
    }
}

pub fn join_stage(
    next_state: UDynVarSys,
    mut output: InfParOutputSys,
    end: BoolVarSys,
) -> InfParOutputSys {
    let state_start = next_state.bitnum();
    let old_state = output.state.clone().subvalue(0, state_start);
    let state_stage = output
        .state
        .clone()
        .subvalue(state_start, output.state.bitnum());
    output.state = dynint_ite(
        end.clone(),
        next_state.concat(UDynVarSys::from_n(0u8, output.state.bitnum() - state_start)),
        old_state.concat(state_stage),
    );
    output
}

// init_mem_address_end_pos - initialize memory address end position from memory.
// Information about MemAddressEndPos in memory:
// At memory address 0: sequences of values between 1..=MAX and one zero,
// MemAddressPosEndPos is sum of non-zero cells.
// If cell_len=1 then: sequences of 1 and one zero. MemAddressPosEndPos is number of 1's.

// function form: f(output_state, UDynVarSys, state_start: usize, in_output: &InfParOutputSys)
//                -> (UDynVarSys, InfParOutputSys)
// arguments:
// output_state - output_state of state_start length that choose this stage
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
    input: &InfParInputSys,
    data_kind: u8,
    dpmove: u8,
    step_num: u64,
) -> (UDynVarSys, InfParOutputSys, BoolVarSys) {
    let state_start = output_state.bitnum();
    let step_num_bits = calc_log_bits_u64(step_num);
    let input_state = extend_output_state(state_start, step_num_bits, input);
    let mut output = InfParOutputSys::new(input.config());
    let end = if step_num_bits != 0 {
        let in_step = input_state.subvalue(state_start, step_num_bits);
        output.state = output_state.clone().concat(&in_step + 1u8);
        (&in_step).equal(UDynVarSys::from_n(step_num - 1, step_num_bits))
    } else {
        output.state = output_state.clone();
        true.into()
    };
    output.dkind = U2VarSys::from(data_kind);
    output.dpmove = U2VarSys::from(dpmove);
    (input_state, output, end)
}

pub fn data_pos_to_start_stage(
    output_state: UDynVarSys,
    input: &InfParInputSys,
    data_kind: u8,
) -> (UDynVarSys, InfParOutputSys, BoolVarSys) {
    let end = !&input.dp_move_done;
    let mut output = InfParOutputSys::new(input.config());
    output.state = output_state;
    output.dkind = U2VarSys::from(data_kind);
    output.dpmove = U2VarSys::from(DPMOVE_BACKWARD);
    (input.state.clone(), output, end)
}

// sequential increase memory address stage -
// sequential - only if all processors have this same memory address.
pub fn seq_increase_mem_address_stage(
    output_state: UDynVarSys,
    input: &InfParInputSys,
) -> (UDynVarSys, InfParOutputSys, BoolVarSys) {
    let state_start = output_state.bitnum();
    // 1. load data part from mem_address.
    // 2. Increase data part value and store to mem_address.
    // 3. If carry after increasing value then:
    // 3.1. Move forward (increase mem_address_pos) in mem_address and go to 1.
    // 4. Otherwise Move mem_address_pos back.
    // 5. If move done then go to 1.
    let input_state = extend_output_state(state_start, 3 + input.dpval.bitnum(), input);
    let (stage, value) = {
        let parts = input_state
            .clone()
            .subvalues(state_start, [3, input.dpval.bitnum()]);
        (
            U3VarSys::try_from(parts[0].clone()).unwrap(),
            parts[1].clone(),
        )
    };
    let output_base = InfParOutputSys::new(input.config());
    // Stage 0b000. 1. load data part from mem_address.
    let out_stage_0 = U3VarSys::from(1u8);
    let out_value_0 = UDynVarSys::from_n(0u8, input.dpval.bitnum());
    let mut output_0 = output_base.clone();
    output_0.state = output_state
        .clone()
        .concat(UDynVarSys::from(out_stage_0))
        .concat(out_value_0);
    // Stage 0b001. 2. Increase data part value and store to mem_address.
    // Stage 0b010. 3. If carry after increasing value then:
    // Stage 0b011. 3.1. Move forward (increase mem_address_pos) in mem_address and go to 1.
    // Stage 0b100. 4. Otherwise Move mem_address_pos back.
    // Stage 0b101. 5. If move done then go to 1.
    (
        extend_output_state(state_start, 4, input),
        InfParOutputSys::new(input.config()),
        true.into(),
    )
}

pub fn init_mem_address_end_pos_stage(
    output_state: UDynVarSys,
    input: &InfParInputSys,
) -> (UDynVarSys, InfParOutputSys, BoolVarSys) {
    let state_start = output_state.bitnum();
    // Stages:
    // 1. Load cell from memory.
    // 2. If cell==0 then end of algorithm.
    // 3. If cell!=0 then increase temp_buffer_pos and decrease this value.
    // 4. If cell==0 then increase memory_address and go to 1.
    let input_state = extend_output_state(state_start, 4, input);
    //let input_state.clone().subvalue(state_start,
    (
        input_state,
        InfParOutputSys::new(input.config()),
        true.into(),
    )
}

// init_proc_id_end_pos - initialize proc id end position from memory.
// Information about ProcIdEndPos in memory: This same as in MemAddressEndPos and start
// after MemAddressEndPos in memory.

pub fn init_proc_id_end_pos_stage(
    output_state: UDynVarSys,
    input: &InfParInputSys,
) -> (UDynVarSys, InfParOutputSys, BoolVarSys) {
    let state_start = output_state.bitnum();
    (
        extend_output_state(state_start, 4, input),
        InfParOutputSys::new(input.config()),
        true.into(),
    )
}
