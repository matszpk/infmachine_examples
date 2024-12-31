use gate_calc_log_bits::*;
use gategen::boolvar::*;
use gategen::dynintvar::*;
use gategen::intvar::*;
use infmachine_config::*;
use infmachine_gen::*;

use rand::random;

use std::env;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

pub mod utils;
use utils::*;

#[derive(Clone, Debug)]
struct PrefixOpState {
    stage: U4VarSys,
    cell: UDynVarSys,
    no_first: BoolVarSys,
    carry: BoolVarSys,
    ext_out: BoolVarSys,
}

// State:
// stage - stage to execute
// cell - loaded memory
// carry - carry from subtraction from memory address (conjunction)
// no_first - if first phase
// ext_output - from shifting temp_buffer[sub]
impl PrefixOpState {
    fn new(cell_len: usize, input_state: &UDynVarSys) -> Self {
        let v = input_state.subvalues(0, [4, cell_len, 1, 1, 1]);
        Self {
            stage: U4VarSys::try_from(v[0].clone()).unwrap(),
            cell: v[1].clone(),
            no_first: v[2].bit(0),
            carry: v[3].bit(0),
            ext_out: v[4].bit(0),
        }
    }
    fn len(cell_len: usize) -> usize {
        4 + cell_len + 3
    }

    fn to_var(self) -> UDynVarSys {
        UDynVarSys::from(self.stage)
            .concat(self.cell)
            .concat(UDynVarSys::from_iter([
                self.no_first,
                self.carry,
                self.ext_out,
            ]))
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
    fn ext_out(mut self, ext_out: BoolVarSys) -> Self {
        self.ext_out = ext_out;
        self
    }
    fn ext_out_pos(&self) -> usize {
        4 + self.cell.bitnum() + 2
    }
}

struct Copy1NAndSet1Func {
    copy1n: Copy1NFunc,
    one0: One0Func,
}

impl Copy1NAndSet1Func {
    fn new(n: usize, len: usize) -> Self {
        Self {
            copy1n: Copy1NFunc::new(n),
            one0: One0Func::new(len),
        }
    }
}

impl FunctionNN for Copy1NAndSet1Func {
    fn state_len(&self) -> usize {
        self.one0.state_len()
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
        let (_, mut out, mut ext_outs) = self.copy1n.output(UDynVarSys::var(0), inputs);
        let (next_state, one_out, one_ext_outs) = self.one0.output(state);
        out.push(one_out);
        ext_outs.extend(one_ext_outs);
        (next_state, out, ext_outs)
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
    let data_part_num = (max_proc_num_bits + data_part_len - 1) / data_part_len;
    let (field_start, temp_buffer_step) = temp_buffer_first_field(data_part_len, 0, 2);
    let orig_field = field_start;
    let sub_field = orig_field + 1;
    let tb_chunk_len = sub_field + 1;
    let mut mobj = InfParMachineObjectSys::new(
        config,
        InfParEnvConfig {
            proc_num,
            flat_memory: true,
            max_mem_size: Some((((proc_num + 32) << cell_len_bits) + 7) >> 3),
            max_temp_buffer_len: (tb_chunk_len + 2) * data_part_num * data_part_len,
        },
    );
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
        0,
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
        Copy1NAndSet1Func::new(2, data_part_len as usize),
    );
    // 3. Load (original) data from memory
    let mut output_3 = InfParOutputSys::new(config);
    output_3.state = input_state.clone().stage_val(4).to_var();
    output_3.memr = true.into();
    // 4. Store to cell
    let mut output_4 = InfParOutputSys::new(config);
    output_4.state = input_state
        .clone()
        .stage_val(5)
        .cell(mach_input.memval.clone())
        .to_var();
    // 5. Do: mem_address = mem_address - temp_buffer[sub]
    //    if carry (if mem_address >= temp_buffer[sub])
    //    state_carry &= carry
    let (output_5, _, ext_out, ext_out_set) =
        par_process_mem_address_temp_buffer_to_mem_address_stage(
            input_state.clone().stage_val(5).to_var(),
            input_state.clone().stage_val(6).to_var(),
            &mut mach_input,
            temp_buffer_step,
            sub_field,
            false,
            Sub2Func::new(),
        );
    let output_5 = install_external_outputs(
        output_5,
        input_state.ext_out_pos(),
        &mach_input.state,
        UDynVarSys::filled(1, ext_out[0].bit(0)),
        ext_out_set,
    );
    // 6. Load memory data to state (arg1). Clear ext_out.
    let mut output_6 = InfParOutputSys::new(config);
    let new_carry = &input_state.carry & &input_state.ext_out;
    output_6.state = input_state
        .clone()
        .stage_val(7)
        .carry(new_carry.clone())
        .ext_out(false.into())
        .to_var();
    output_6.memr = new_carry;
    // 7. If state_carry: cell = cell + arg1.
    let mut output_7 = InfParOutputSys::new(config);
    output_7.state = input_state
        .clone()
        .stage_val(8)
        .cell(
            // if state_carry == 1 then do it
            dynint_ite(
                input_state.carry.clone(),
                op(input_state.cell.clone(), mach_input.memval.clone()),
                input_state.cell.clone(),
            ),
        )
        .to_var();
    // 8. Swap temp_buffer[orig] and mem_address.
    let (output_8, _, _, _) = par_process_infinite_data_stage(
        input_state.clone().stage_val(8).to_var(),
        input_state.clone().stage_val(9).to_var(),
        &mut mach_input,
        temp_buffer_step,
        &[
            (InfDataParam::MemAddress, END_POS_MEM_ADDRESS),
            (InfDataParam::TempBuffer(orig_field), END_POS_MEM_ADDRESS),
        ],
        &[
            (InfDataParam::MemAddress, END_POS_MEM_ADDRESS),
            (InfDataParam::TempBuffer(orig_field), END_POS_MEM_ADDRESS),
        ],
        FuncNNAdapter2_2::from(Swap2Func::new()),
    );
    // 9. Store cell to memory.
    let mut output_9 = InfParOutputSys::new(config);
    output_9.state = input_state.clone().stage_val(10).to_var();
    output_9.memw = input_state.carry.clone();
    output_9.memval = input_state.cell.clone();
    // 10. Swap temp_buffer[orig] and mem_address.
    let (output_10, _, _, _) = par_process_infinite_data_stage(
        input_state.clone().stage_val(10).to_var(),
        input_state
            .clone()
            .stage(int_ite(
                input_state.no_first.clone(),
                // do next step
                U4VarSys::from(11u8),
                // skip next step if first
                U4VarSys::from(12u8),
            ))
            .to_var(),
        &mut mach_input,
        temp_buffer_step,
        &[
            (InfDataParam::MemAddress, END_POS_MEM_ADDRESS),
            (InfDataParam::TempBuffer(orig_field), END_POS_MEM_ADDRESS),
        ],
        &[
            (InfDataParam::MemAddress, END_POS_MEM_ADDRESS),
            (InfDataParam::TempBuffer(orig_field), END_POS_MEM_ADDRESS),
        ],
        FuncNNAdapter2_2::from(Swap2Func::new()),
    );
    // 11. If no_first: temp_buffer[sub] <<= 1.
    let (output_11, _, ext_out, ext_out_set) = par_process_temp_buffer_to_temp_buffer_stage(
        input_state.clone().stage_val(11).to_var(),
        input_state.clone().stage_val(12).to_var(),
        &mut mach_input,
        temp_buffer_step,
        sub_field,
        sub_field,
        false,
        false,
        Shl1Func::new(data_part_len as usize, 1),
    );
    let output_11 = install_external_outputs(
        output_11,
        input_state.ext_out_pos(),
        &mach_input.state,
        UDynVarSys::filled(1, ext_out[0].bit(0)),
        ext_out_set,
    );
    // 12. Set no_first = 1.
    //     Check if temp_buffer[sub] = end: if yes then: end otherwise go to 5.
    let mut output_12 = InfParOutputSys::new(config);
    output_12.state = input_state
        .clone()
        .stage_val(5)
        .no_first(true.into())
        .to_var();
    output_12.stop = input_state.ext_out.clone();
    finish_machine_with_table(
        mobj,
        &mach_input,
        vec![
            output_0, output_1, output_2, output_3, output_4, output_5, output_6, output_7,
            output_8, output_9, output_10, output_11, output_12,
        ],
        input_state.stage.into(),
    )
    .to_toml()
}

fn gen_data_and_expected(
    cell_len_bits: u32,
    data_part_len: u32,
    proc_num: u64,
    max_value: u64,
    data_path: impl AsRef<Path>,
    expected_path: Option<impl AsRef<Path>>,
    op: impl Fn(u64, u64) -> u64,
    init_value: u64,
) -> std::io::Result<()> {
    let mut data_writer = CellWriter::new(cell_len_bits, BufWriter::new(File::create(data_path)?));
    let mut expected_writer_opt = if let Some(path) = expected_path {
        Some(CellWriter::new(
            cell_len_bits,
            BufWriter::new(File::create(path)?),
        ))
    } else {
        None
    };
    let mem_num_bits = u32::try_from(calc_log_bits_u64(proc_num + 32)).unwrap();
    let proc_num_bits = u32::try_from(calc_log_bits_u64(proc_num)).unwrap();
    mem_address_proc_id_setup(
        &mut data_writer,
        0,
        ((mem_num_bits + data_part_len - 1) / data_part_len) as u64,
        ((proc_num_bits + data_part_len - 1) / data_part_len) as u64,
    )?;
    let mut cum_v = init_value;
    let cell_mask = if cell_len_bits < 6 {
        (1u64 << (1 << cell_len_bits)) - 1
    } else {
        u64::MAX
    };
    for _ in 0..proc_num {
        let v = random::<u64>() % max_value;
        data_writer.write_cell(v)?;
        cum_v = op(cum_v, v) & cell_mask;
        if let Some(exp_writer) = expected_writer_opt.as_mut() {
            exp_writer.write_cell(cum_v)?;
        }
    }
    Ok(())
}

fn main() {
    let mut args = env::args();
    args.next().unwrap();
    let command = args.next().unwrap();
    let op = args.next().unwrap();
    let cell_len_bits: u32 = args.next().unwrap().parse().unwrap();
    let data_part_len: u32 = args.next().unwrap().parse().unwrap();
    let proc_num: u64 = args.next().unwrap().parse().unwrap();
    assert!(cell_len_bits <= 16);
    assert_ne!(data_part_len, 0);
    assert_ne!(proc_num, 0);
    match command.as_str() {
        "machine" => {
            let max_proc_num_bits: u32 = if let Some(arg) = args.next() {
                arg.parse().unwrap()
            } else {
                u32::try_from(calc_log_bits_u64(proc_num)).unwrap()
            };
            assert_ne!(max_proc_num_bits, 0);
            assert!(max_proc_num_bits <= 64);
            assert!(u128::from(proc_num) <= (1u128 << max_proc_num_bits));
            print!(
                "{}",
                callsys(|| {
                    gen_prefix_op(
                        cell_len_bits,
                        data_part_len,
                        proc_num,
                        max_proc_num_bits,
                        match op.as_str() {
                            "add" => |arg1, arg2| arg1 + arg2,
                            "mul1" => |arg1: UDynVarSys, arg2: UDynVarSys| {
                                ((arg1 + 1u8) * (arg2 + 1u8)) - 1u8
                            },
                            "and" => |arg1, arg2| arg1 & arg2,
                            "or" => |arg1, arg2| arg1 | arg2,
                            "xor" => |arg1, arg2| arg1 ^ arg2,
                            "min" => |arg1, arg2| dynint_min(arg1, arg2),
                            "max" => |arg1, arg2| dynint_max(arg1, arg2),
                            _ => {
                                panic!("Unknown op");
                            }
                        },
                    )
                    .unwrap()
                })
            );
        }
        "data_and_exp" => {
            assert!(cell_len_bits <= 6);
            let max_value: u64 = args.next().unwrap().parse().unwrap();
            let data_path = args.next().unwrap();
            let expected_path = args.next();
            gen_data_and_expected(
                cell_len_bits,
                data_part_len,
                proc_num,
                max_value,
                data_path,
                expected_path,
                match op.as_str() {
                    "add" => |arg1: u64, arg2: u64| arg1.overflowing_add(arg2).0,
                    "mul1" => |arg1: u64, arg2: u64| {
                        (arg1.overflowing_add(1).0)
                            .overflowing_mul(arg2.overflowing_add(1).0)
                            .0
                            .overflowing_sub(1)
                            .0
                    },
                    "and" => |arg1, arg2| arg1 & arg2,
                    "or" => |arg1, arg2| arg1 | arg2,
                    "xor" => |arg1, arg2| arg1 ^ arg2,
                    "min" => |arg1, arg2| std::cmp::min(arg1, arg2),
                    "max" => |arg1, arg2| std::cmp::max(arg1, arg2),
                    _ => {
                        panic!("Unknown op");
                    }
                },
                match op.as_str() {
                    "add" => 0,
                    "mul1" => 0,
                    "and" => u64::MAX,
                    "or" => 0,
                    "xor" => 0,
                    "min" => u64::MAX,
                    "max" => 0,
                    _ => {
                        panic!("Unknown op");
                    }
                },
            )
            .unwrap();
        }
        _ => {
            panic!("Unknown command");
        }
    }
}
