use gategen::boolvar::*;
use gategen::dynintvar::*;
use gategen::intvar::*;
use infmachine_config::*;
use infmachine_gen::*;

use std::env;

pub mod utils;
use utils::*;

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

#[derive(Clone, Debug)]
struct PrefixOpState {
    stage: U4VarSys,
    cell: UDynVarSys,
    no_first: BoolVarSys,
    carry: BoolVarSys,
    end: BoolVarSys,
}

// State:
// stage - stage to execute
// cell - loaded memory
// carry - carry from subtraction from memory address (conjunction)
// no_first - if first phase
// ext_output - from shifting temp_buffer[sub]
impl PrefixOpState {
    fn new(cell_len: usize, input_state: &UDynVarSys) -> Self {
        let v = input_state.subvalues(0, [cell_len, 1, 1, 1]);
        Self {
            stage: U4VarSys::try_from(v[0].clone()).unwrap(),
            cell: v[1].clone(),
            no_first: v[2].bit(0),
            carry: v[3].bit(0),
            end: v[4].bit(0),
        }
    }
    fn len(cell_len: usize) -> usize {
        4 + cell_len + 3
    }

    fn to_var(self) -> UDynVarSys {
        UDynVarSys::from(self.stage)
            .concat(self.cell)
            .concat(UDynVarSys::from_iter([self.no_first, self.carry, self.end]))
    }

    fn stage(mut self, stage: U4VarSys) -> Self {
        self.stage = stage;
        self
    }
    fn stage_val(mut self, stage: usize) -> Self {
        self.stage = stage.into();
        self
    }
    fn cell(mut self, cell: UDynVarSys) -> Self {
        self.cell = cell;
        self
    }
    fn no_first(mut self, no_first: BoolVarSys) -> Self {
        self.no_first = no_first;
        self
    }
    fn carry(mut self, carry: BoolVarSys) -> Self {
        self.carry = carry;
        self
    }
    fn end(mut self, end: BoolVarSys) -> Self {
        self.end = end;
        self
    }
}

struct Copy1NAndSet1Func {
    copy1n: Copy1NFunc,
}

impl Copy1NAndSet1Func {
    fn new(n: usize) -> Self {
        Self {
            copy1n: Copy1NFunc::new(n),
        }
    }
}

impl FunctionNN for Copy1NAndSet1Func {
    fn state_len(&self) -> usize {
        1
    }
    fn input_num(&self) -> usize {
        self.copy1n.input_num()
    }
    fn output_num(&self) -> usize {
        self.copy1n.output_num() + 1
    }
    fn output(
        &self,
        state: UDynVarSys,
        inputs: &[UDynVarSys],
    ) -> (UDynVarSys, Vec<UDynVarSys>, Vec<UDynVarSys>) {
        let len = inputs[0].bitnum();
        let (_, mut out, ext_outs) = self.copy1n.output(UDynVarSys::var(0), inputs);
        out.push(dynint_ite(
            !state.bit(0),
            UDynVarSys::from_n(1u8, len),
            UDynVarSys::from_n(0u8, len),
        ));
        (UDynVarSys::from_n(1u8, 1), out, vec![])
    }
}

fn gen_prefix_op(
    cell_len_bits: u32,
    data_part_len: u32,
    proc_num: u64,
    max_proc_num_bits: u32,
    op: impl Fn(UDynVarSys, UDynVarSys) -> UDynVarSys,
) -> Result<String, toml::ser::Error> {
    let config = InfParInterfaceConfig {
        cell_len_bits,
        data_part_len,
    };
    let cell_len = 1 << cell_len_bits;
    let mut mobj = InfParMachineObjectSys::new(
        config,
        InfParEnvConfig {
            proc_num,
            flat_memory: true,
            max_mem_size: Some(((proc_num << cell_len_bits) + 7) >> 3),
            max_temp_buffer_len: max_proc_num_bits,
        },
    );
    let (field_start, temp_buffer_step) = temp_buffer_first_field(data_part_len, 0, 2);
    let orig_field = field_start;
    let sub_field = orig_field + 1;
    mobj.in_state = Some(UDynVarSys::var(PrefixOpState::len(cell_len)));
    let mut mach_input = mobj.input();
    let input_state = PrefixOpState::new(cell_len, &mach_input.state);
    // Main stages:
    // no_first = 0 - in state.
    // 0. Init memory and proc end pos.
    let (output_0, _) = init_machine_end_pos_stage(
        input_state.clone().stage_val(0).to_var(),
        input_state.clone().stage_val(1).to_var(),
        &mut mach_input,
        temp_buffer_step,
    );
    // 1. Move mem data to start.
    let (output_1, _) = mem_data_to_start(
        input_state.clone().stage_val(1).to_var(),
        input_state.clone().stage_val(2).to_var(),
        &mut mach_input,
        temp_buffer_step,
        1,
    );
    // 2. Initialize memory address = proc_id, temp_buffer[orig] = proc_id.
    //    Initialize temp_buffer[sub] = 1. State_carry = 1.
    let (output_2, _, _, _) = par_process_infinite_data_stage(
        input_state.clone().stage_val(2).to_var(),
        input_state.clone().stage_val(3).carry(true.into()).to_var(),
        &mut mach_input,
        temp_buffer_step,
        &[(InfDataParam::ProcId, END_POS_PROC_ID)],
        &[
            (InfDataParam::MemAddress, END_POS_MEM_ADDRESS),
            (InfDataParam::TempBuffer(orig_field), END_POS_MEM_ADDRESS),
            (InfDataParam::TempBuffer(sub_field), END_POS_MEM_ADDRESS),
        ],
        Copy1NAndSet1Func::new(2),
    );
    // 3. Load data from memory.
    // 4. Do: mem_address = mem_address - temp_buffer[sub]
    //    if carry (if mem_address >= temp_buffer[sub])
    //    state_carry &= carry
    // 5. Load memory data to state (arg1).
    // 6. If state_carry: cell = cell + arg1.
    // 7. Swap temp_buffer[orig] and mem_address.
    // 8. Store cell to memory.
    // 9. Swap temp_buffer[orig] and mem_address.
    // 10. If not no_first: temp_buffer[sub] <<= 1.
    // 11. Set no_first = 1.
    //     Check if temp_buffer[sub] = end: if yes then: end otherwise go to 3.
    mobj.to_machine().to_toml()
}

fn main() {
    let mut args = env::args();
    args.next().unwrap();
    let cell_len_bits: u32 = args.next().unwrap().parse().unwrap();
    let data_part_len: u32 = args.next().unwrap().parse().unwrap();
    let proc_num: u64 = args.next().unwrap().parse().unwrap();
    let max_proc_num_bits: u32 = if let Some(arg) = args.next() {
        arg.parse().unwrap()
    } else {
        u32::try_from(calc_log_bits_u64(proc_num)).unwrap()
    };
    assert!(cell_len_bits <= 16);
    assert_ne!(data_part_len, 0);
    assert_ne!(proc_num, 0);
    assert_ne!(max_proc_num_bits, 0);
    assert!((1 << cell_len_bits) < max_proc_num_bits);
    assert!(max_proc_num_bits <= 64);
    assert!(u128::from(proc_num) <= (1u128 << max_proc_num_bits));
    print!(
        "{}",
        callsys(|| gen_prefix_op(
            cell_len_bits,
            data_part_len,
            proc_num,
            max_proc_num_bits,
            |arg1, arg2| arg1 + arg2
        )
        .unwrap())
    );
}
