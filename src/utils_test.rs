use gategen::boolvar::*;
use gategen::dynintvar::*;
use gategen::intvar::*;
use infmachine_config::*;
use infmachine_gen::*;

use std::env;

pub mod utils;
use utils::*;

fn gen_move_data_pos_test(
    cell_len_bits: u32,
    data_part_len: u32,
    temp_buffer_len: u32,
    proc_num: u64,
    mem_size: u64,
    step_num: u64,
) -> Result<String, toml::ser::Error> {
    let config = InfParInterfaceConfig {
        cell_len_bits,
        data_part_len,
    };
    let mut mobj = InfParMachineObjectSys::new(
        config,
        InfParEnvConfig {
            proc_num,
            flat_memory: true,
            max_mem_size: Some(mem_size),
            max_temp_buffer_len: temp_buffer_len,
        },
    );
    mobj.in_state = Some(UDynVarSys::var(2));
    let mut mach_input = mobj.input();
    let unused_bit = UDynVarSys::filled(1, unused_inputs(&mobj, mach_input.state.bit(1)));
    // first stage
    let (output_1, _) = move_data_pos_stage(
        UDynVarSys::from_n(0u8, 1).concat(unused_bit.clone()),
        UDynVarSys::from_n(1u8, 1).concat(unused_bit.clone()),
        &mut mach_input,
        DKIND_TEMP_BUFFER,
        DPMOVE_FORWARD,
        step_num,
    );
    // stop stage
    let mut output_2 = InfParOutputSys::new(config);
    output_2.state = UDynVarSys::from_n(1u8, 1).concat(unused_bit.clone());
    output_2.stop = true.into();
    let mut output_stages = vec![output_1, output_2];
    InfParOutputSys::fix_state_len(&mut output_stages);
    let final_state = dynint_table(
        mach_input.state.clone().subvalue(0, 1),
        output_stages.into_iter().map(|v| v.to_dynintvar()),
    );
    mobj.in_state = Some(mach_input.state);
    mobj.from_dynintvar(final_state);
    mobj.to_machine().to_toml()
}

fn gen_move_data_pos_and_back_test(
    cell_len_bits: u32,
    data_part_len: u32,
    temp_buffer_len: u32,
    proc_num: u64,
    mem_size: u64,
    step_num: u64,
) -> Result<String, toml::ser::Error> {
    let config = InfParInterfaceConfig {
        cell_len_bits,
        data_part_len,
    };
    let mut mobj = InfParMachineObjectSys::new(
        config,
        InfParEnvConfig {
            proc_num,
            flat_memory: true,
            max_mem_size: Some(mem_size),
            max_temp_buffer_len: temp_buffer_len,
        },
    );
    mobj.in_state = Some(UDynVarSys::var(3));
    let mut mach_input = mobj.input();
    let unused_bit = UDynVarSys::filled(1, unused_inputs(&mobj, mach_input.state.bit(0)));
    // first stage
    let (output_1, _) = move_data_pos_stage(
        unused_bit.clone().concat(UDynVarSys::from_n(0u8, 2)),
        unused_bit.clone().concat(UDynVarSys::from_n(1u8, 2)),
        &mut mach_input,
        DKIND_TEMP_BUFFER,
        DPMOVE_FORWARD,
        step_num,
    );
    // back stage
    let (output_2, _) = data_pos_to_start_stage(
        unused_bit.clone().concat(UDynVarSys::from_n(1u8, 2)),
        unused_bit.clone().concat(UDynVarSys::from_n(2u8, 2)),
        &mut mach_input,
        DKIND_TEMP_BUFFER,
    );
    // stop stage
    let mut output_3 = InfParOutputSys::new(config);
    output_3.state = unused_bit.concat(UDynVarSys::from_n(2u8, 2));
    output_3.stop = true.into();

    let mut output_stages = vec![output_1, output_2, output_3.clone(), output_3];
    InfParOutputSys::fix_state_len(&mut output_stages);
    let final_state = dynint_table(
        mach_input.state.clone().subvalue(1, 2),
        output_stages.into_iter().map(|v| v.to_dynintvar()),
    );
    mobj.in_state = Some(mach_input.state);
    mobj.from_dynintvar(final_state);
    mobj.to_machine().to_toml()
}

fn main() {
    let mut args = env::args();
    args.next().unwrap();
    let stage = args.next().unwrap();
    let cell_len_bits: u32 = args.next().unwrap().parse().unwrap();
    let data_part_len: u32 = args.next().unwrap().parse().unwrap();
    let temp_buffer_len: u32 = args.next().unwrap().parse().unwrap();
    let proc_num: u64 = args.next().unwrap().parse().unwrap();
    let mem_size: u64 = args.next().unwrap().parse().unwrap();
    assert!(cell_len_bits <= 16);
    assert_ne!(data_part_len, 0);
    assert_ne!(proc_num, 0);
    assert_ne!(mem_size, 0);
    match stage.as_str() {
        "move_data_pos" => {
            let step_num: u64 = args.next().unwrap().parse().unwrap();
            assert_ne!(step_num, 0);
            print!(
                "{}",
                callsys(|| gen_move_data_pos_test(
                    cell_len_bits,
                    data_part_len,
                    temp_buffer_len,
                    proc_num,
                    mem_size,
                    step_num,
                )
                .unwrap())
            );
        }
        "move_data_pos_and_back" => {
            let step_num: u64 = args.next().unwrap().parse().unwrap();
            assert_ne!(step_num, 0);
            print!(
                "{}",
                callsys(|| gen_move_data_pos_and_back_test(
                    cell_len_bits,
                    data_part_len,
                    temp_buffer_len,
                    proc_num,
                    mem_size,
                    step_num,
                )
                .unwrap())
            );
        }
        _ => {
            panic!("Unknown example");
        }
    }
}
