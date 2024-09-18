use gategen::boolvar::*;
use gategen::dynintvar::*;
use infmachine_config::*;
use infmachine_gen::*;

use std::env;

pub mod utils;
use utils::*;

const fn calc_log_bits_u64(n: u64) -> usize {
    let nbits = u64::BITS - n.leading_zeros();
    if (1 << (nbits - 1)) == n {
        (nbits - 1) as usize
    } else {
        nbits as usize
    }
}

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

fn gen_move_data_pos_expr_test(
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
    let (output_1, _) = move_data_pos_expr_stage(
        UDynVarSys::from_n(0u8, 1).concat(unused_bit.clone()),
        UDynVarSys::from_n(1u8, 1).concat(unused_bit.clone()),
        &mut mach_input,
        DKIND_TEMP_BUFFER,
        DPMOVE_FORWARD,
        UDynVarSys::from_n(step_num.checked_sub(1).unwrap(), 64),
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

fn gen_inc_mem_address_test(
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
    let step_num_bits = calc_log_bits_u64(step_num);
    mobj.in_state = Some(UDynVarSys::var(1 + step_num_bits));
    let mut mach_input = mobj.input();
    let unused_bit = UDynVarSys::filled(1, unused_inputs(&mobj, mach_input.state.bit(0)));
    // first stage
    let in_step_count = mach_input.state.subvalue(1, step_num_bits);
    let (mut output_1, end_1) = seq_increase_mem_address_stage(
        unused_bit.clone().concat(in_step_count.clone()),
        unused_bit.clone().concat(&in_step_count + 1u8),
        &mut mach_input,
    );
    output_1.stop = end_1 & in_step_count.equal(step_num - 1);

    mobj.in_state = Some(mach_input.state);
    mobj.from_output(output_1);
    mobj.to_machine().to_toml()
}

fn gen_init_machine_end_pos_one_test(
    cell_len_bits: u32,
    data_part_len: u32,
    temp_buffer_len: u32,
    proc_num: u64,
    mem_size: u64,
    temp_buffer_step: u32,
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
    mobj.in_state = Some(UDynVarSys::var(1));
    let mut mach_input = mobj.input();
    // first stage
    let (output_1, _) = init_machine_end_pos_stage(
        UDynVarSys::from_n(0u8, 1),
        UDynVarSys::from_n(1u8, 1),
        &mut mach_input,
        temp_buffer_step,
    );
    // stop stage
    let mut output_2 = InfParOutputSys::new(config);
    output_2.state = mach_input.state.clone();
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

fn gen_copy_proc_id_to_temp_buffer_test(
    cell_len_bits: u32,
    data_part_len: u32,
    temp_buffer_len: u32,
    proc_num: u64,
    mem_size: u64,
    temp_buffer_step: u32,
    temp_buffer_step_pos: u32,
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
    // first stage
    let (output_1, _) = init_machine_end_pos_stage(
        UDynVarSys::from_n(0u8, 2),
        UDynVarSys::from_n(1u8, 2),
        &mut mach_input,
        temp_buffer_step,
    );
    let (output_2, _) = par_copy_proc_id_to_temp_buffer_stage(
        UDynVarSys::from_n(1u8, 2),
        UDynVarSys::from_n(2u8, 2),
        &mut mach_input,
        temp_buffer_step,
        temp_buffer_step_pos,
    );
    // stop stage
    let mut output_3 = InfParOutputSys::new(config);
    output_3.state = mach_input.state.clone();
    output_3.stop = true.into();
    let mut output_stages = vec![output_1, output_2, output_3.clone(), output_3];
    InfParOutputSys::fix_state_len(&mut output_stages);
    let final_state = dynint_table(
        mach_input.state.clone().subvalue(0, 2),
        output_stages.into_iter().map(|v| v.to_dynintvar()),
    );
    mobj.in_state = Some(mach_input.state);
    mobj.from_dynintvar(final_state);
    mobj.to_machine().to_toml()
}

fn gen_copy_proc_id_to_mem_address_test(
    cell_len_bits: u32,
    data_part_len: u32,
    temp_buffer_len: u32,
    proc_num: u64,
    mem_size: u64,
    temp_buffer_step: u32,
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
    // first stage
    let (output_1, _) = init_machine_end_pos_stage(
        UDynVarSys::from_n(0u8, 2),
        UDynVarSys::from_n(1u8, 2),
        &mut mach_input,
        temp_buffer_step,
    );
    let (output_2, _) = par_copy_proc_id_to_mem_address_stage(
        UDynVarSys::from_n(1u8, 2),
        UDynVarSys::from_n(2u8, 2),
        &mut mach_input,
        temp_buffer_step,
    );
    // stop stage
    let mut output_3 = InfParOutputSys::new(config);
    output_3.state = mach_input.state.clone();
    output_3.stop = true.into();
    let mut output_stages = vec![output_1, output_2, output_3.clone(), output_3];
    InfParOutputSys::fix_state_len(&mut output_stages);
    let final_state = dynint_table(
        mach_input.state.clone().subvalue(0, 2),
        output_stages.into_iter().map(|v| v.to_dynintvar()),
    );
    mobj.in_state = Some(mach_input.state);
    mobj.from_dynintvar(final_state);
    mobj.to_machine().to_toml()
}

fn gen_copy_proc_id_to_mem_address_test_2(
    cell_len_bits: u32,
    data_part_len: u32,
    temp_buffer_len: u32,
    proc_num: u64,
    mem_size: u64,
    temp_buffer_step: u32,
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
    // first stage
    let (output_1, _) = init_machine_end_pos_stage(
        UDynVarSys::from_n(0u8, 2),
        UDynVarSys::from_n(1u8, 2),
        &mut mach_input,
        temp_buffer_step,
    );
    let (output_2, _) = par_copy_proc_id_to_mem_address_stage(
        UDynVarSys::from_n(1u8, 2),
        UDynVarSys::from_n(2u8, 2),
        &mut mach_input,
        temp_buffer_step,
    );
    // stop stage
    let mut output_3 = InfParOutputSys::new(config);
    output_3.state = mach_input.state.clone();
    output_3.stop = true.into();
    finish_machine_with_table(
        mobj,
        &mach_input,
        vec![output_1, output_2, output_3],
        mach_input.state.clone().subvalue(0, 2),
    )
    .to_toml()
}

fn gen_copy_temp_buffer_to_mem_address_test(
    cell_len_bits: u32,
    data_part_len: u32,
    temp_buffer_len: u32,
    proc_num: u64,
    mem_size: u64,
    temp_buffer_step: u32,
    temp_buffer_step_pos: u32,
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
    // first stage
    let (output_1, _) = init_machine_end_pos_stage(
        UDynVarSys::from_n(0u8, 2),
        UDynVarSys::from_n(1u8, 2),
        &mut mach_input,
        temp_buffer_step,
    );
    let (output_2, _) = par_copy_proc_id_to_temp_buffer_stage(
        UDynVarSys::from_n(1u8, 2),
        UDynVarSys::from_n(2u8, 2),
        &mut mach_input,
        temp_buffer_step,
        temp_buffer_step_pos,
    );
    let (output_3, _) = par_copy_temp_buffer_to_mem_address_stage(
        UDynVarSys::from_n(2u8, 2),
        UDynVarSys::from_n(3u8, 2),
        &mut mach_input,
        temp_buffer_step,
        temp_buffer_step_pos,
    );
    // stop stage
    let mut output_4 = InfParOutputSys::new(config);
    output_4.state = mach_input.state.clone();
    output_4.stop = true.into();
    let mut output_stages = vec![output_1, output_2, output_3, output_4];
    InfParOutputSys::fix_state_len(&mut output_stages);
    let final_state = dynint_table(
        mach_input.state.clone().subvalue(0, 2),
        output_stages.into_iter().map(|v| v.to_dynintvar()),
    );
    mobj.in_state = Some(mach_input.state);
    mobj.from_dynintvar(final_state);
    mobj.to_machine().to_toml()
}

fn gen_copy_mem_address_to_temp_buffer_test(
    cell_len_bits: u32,
    data_part_len: u32,
    temp_buffer_len: u32,
    proc_num: u64,
    mem_size: u64,
    temp_buffer_step: u32,
    temp_buffer_step_pos: u32,
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
    // first stage
    let (output_1, _) = init_machine_end_pos_stage(
        UDynVarSys::from_n(0u8, 2),
        UDynVarSys::from_n(1u8, 2),
        &mut mach_input,
        temp_buffer_step,
    );
    let (output_2, _) = par_copy_proc_id_to_mem_address_stage(
        UDynVarSys::from_n(1u8, 2),
        UDynVarSys::from_n(2u8, 2),
        &mut mach_input,
        temp_buffer_step,
    );
    let (output_3, _) = par_copy_mem_address_to_temp_buffer_stage(
        UDynVarSys::from_n(2u8, 2),
        UDynVarSys::from_n(3u8, 2),
        &mut mach_input,
        temp_buffer_step,
        temp_buffer_step_pos,
    );
    // stop stage
    let mut output_4 = InfParOutputSys::new(config);
    output_4.state = mach_input.state.clone();
    output_4.stop = true.into();
    let mut output_stages = vec![output_1, output_2, output_3, output_4];
    InfParOutputSys::fix_state_len(&mut output_stages);
    let final_state = dynint_table(
        mach_input.state.clone().subvalue(0, 2),
        output_stages.into_iter().map(|v| v.to_dynintvar()),
    );
    mobj.in_state = Some(mach_input.state);
    mobj.from_dynintvar(final_state);
    mobj.to_machine().to_toml()
}

fn gen_copy_temp_buffer_to_temp_buffer_test(
    cell_len_bits: u32,
    data_part_len: u32,
    temp_buffer_len: u32,
    proc_num: u64,
    mem_size: u64,
    temp_buffer_step: u32,
    tbs_src_pos: u32,
    tbs_dest_pos: u32,
    proc_id_end_pos: bool,
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
    // first stage
    let (output_1, _) = init_machine_end_pos_stage(
        UDynVarSys::from_n(0u8, 2),
        UDynVarSys::from_n(1u8, 2),
        &mut mach_input,
        temp_buffer_step,
    );
    let (output_2, _) = par_copy_proc_id_to_temp_buffer_stage(
        UDynVarSys::from_n(1u8, 2),
        UDynVarSys::from_n(2u8, 2),
        &mut mach_input,
        temp_buffer_step,
        tbs_src_pos,
    );
    let (output_3, _) = par_copy_temp_buffer_to_temp_buffer_stage(
        UDynVarSys::from_n(2u8, 2),
        UDynVarSys::from_n(3u8, 2),
        &mut mach_input,
        temp_buffer_step,
        tbs_src_pos,
        tbs_dest_pos,
        proc_id_end_pos,
    );
    // stop stage
    let mut output_4 = InfParOutputSys::new(config);
    output_4.state = mach_input.state.clone();
    output_4.stop = true.into();
    let mut output_stages = vec![output_1, output_2, output_3, output_4];
    InfParOutputSys::fix_state_len(&mut output_stages);
    let final_state = dynint_table(
        mach_input.state.clone().subvalue(0, 2),
        output_stages.into_iter().map(|v| v.to_dynintvar()),
    );
    mobj.in_state = Some(mach_input.state);
    mobj.from_dynintvar(final_state);
    mobj.to_machine().to_toml()
}

fn gen_process_proc_id_to_temp_buffer_test(
    cell_len_bits: u32,
    data_part_len: u32,
    temp_buffer_len: u32,
    proc_num: u64,
    mem_size: u64,
    temp_buffer_step: u32,
    temp_buffer_step_pos: u32,
    func: impl Function1,
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
    // first stage
    let (output_1, _) = init_machine_end_pos_stage(
        UDynVarSys::from_n(0u8, 2),
        UDynVarSys::from_n(1u8, 2),
        &mut mach_input,
        temp_buffer_step,
    );
    let (output_2, _, _, _) = par_process_proc_id_to_temp_buffer_stage(
        UDynVarSys::from_n(1u8, 2),
        UDynVarSys::from_n(2u8, 2),
        &mut mach_input,
        temp_buffer_step,
        temp_buffer_step_pos,
        func,
    );
    // stop stage
    let mut output_3 = InfParOutputSys::new(config);
    output_3.state = mach_input.state.clone();
    output_3.stop = true.into();
    let mut output_stages = vec![output_1, output_2, output_3.clone(), output_3];
    InfParOutputSys::fix_state_len(&mut output_stages);
    let final_state = dynint_table(
        mach_input.state.clone().subvalue(0, 2),
        output_stages.into_iter().map(|v| v.to_dynintvar()),
    );
    mobj.in_state = Some(mach_input.state);
    mobj.from_dynintvar(final_state);
    mobj.to_machine().to_toml()
}

fn gen_process_proc_id_to_mem_address_test(
    cell_len_bits: u32,
    data_part_len: u32,
    temp_buffer_len: u32,
    proc_num: u64,
    mem_size: u64,
    temp_buffer_step: u32,
    func: impl Function1,
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
    // first stage
    let (output_1, _) = init_machine_end_pos_stage(
        UDynVarSys::from_n(0u8, 2),
        UDynVarSys::from_n(1u8, 2),
        &mut mach_input,
        temp_buffer_step,
    );
    let (output_2, _, _, _) = par_process_proc_id_to_mem_address_stage(
        UDynVarSys::from_n(1u8, 2),
        UDynVarSys::from_n(2u8, 2),
        &mut mach_input,
        temp_buffer_step,
        func,
    );
    // stop stage
    let mut output_3 = InfParOutputSys::new(config);
    output_3.state = mach_input.state.clone();
    output_3.stop = true.into();
    let mut output_stages = vec![output_1, output_2, output_3.clone(), output_3];
    InfParOutputSys::fix_state_len(&mut output_stages);
    let final_state = dynint_table(
        mach_input.state.clone().subvalue(0, 2),
        output_stages.into_iter().map(|v| v.to_dynintvar()),
    );
    mobj.in_state = Some(mach_input.state);
    mobj.from_dynintvar(final_state);
    mobj.to_machine().to_toml()
}

fn gen_process_proc_id_to_mem_address_ext_out_test(
    cell_len_bits: u32,
    data_part_len: u32,
    temp_buffer_len: u32,
    proc_num: u64,
    mem_size: u64,
    temp_buffer_step: u32,
    ext_out_index: usize,
    ext_out_bit: usize,
    func: impl Function1,
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
    // first stage
    let (output_1, _) = init_machine_end_pos_stage(
        UDynVarSys::from_n(0u8, 2).concat(UDynVarSys::filled(1, mach_input.state.bit(2))),
        UDynVarSys::from_n(1u8, 2).concat(UDynVarSys::filled(1, mach_input.state.bit(2))),
        &mut mach_input,
        temp_buffer_step,
    );
    let (mut output_2, _, ext_out, ext_out_set) = par_process_proc_id_to_mem_address_stage(
        UDynVarSys::from_n(1u8, 2).concat(UDynVarSys::filled(1, mach_input.state.bit(2))),
        UDynVarSys::from_n(2u8, 2).concat(UDynVarSys::filled(1, mach_input.state.bit(2))),
        &mut mach_input,
        temp_buffer_step,
        func,
    );
    output_2 = install_external_outputs(
        output_2,
        2,
        &mach_input.state,
        UDynVarSys::filled(1, ext_out[ext_out_index].bit(ext_out_bit)),
        ext_out_set,
    );
    // stop stage
    let mut output_3 = InfParOutputSys::new(config);
    output_3.state = mach_input.state.clone();
    output_3.stop = true.into();
    let mut output_stages = vec![output_1, output_2, output_3.clone(), output_3];
    InfParOutputSys::fix_state_len(&mut output_stages);
    let final_state = dynint_table(
        mach_input.state.clone().subvalue(0, 2),
        output_stages.into_iter().map(|v| v.to_dynintvar()),
    );
    mobj.in_state = Some(mach_input.state);
    mobj.from_dynintvar(final_state);
    mobj.to_machine().to_toml()
}

fn gen_process_temp_buffer_to_mem_address_test(
    cell_len_bits: u32,
    data_part_len: u32,
    temp_buffer_len: u32,
    proc_num: u64,
    mem_size: u64,
    temp_buffer_step: u32,
    temp_buffer_step_pos: u32,
    func: impl Function1,
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
    // first stage
    let (output_1, _) = init_machine_end_pos_stage(
        UDynVarSys::from_n(0u8, 2),
        UDynVarSys::from_n(1u8, 2),
        &mut mach_input,
        temp_buffer_step,
    );
    let (output_2, _) = par_copy_proc_id_to_temp_buffer_stage(
        UDynVarSys::from_n(1u8, 2),
        UDynVarSys::from_n(2u8, 2),
        &mut mach_input,
        temp_buffer_step,
        temp_buffer_step_pos,
    );
    let (output_3, _, _, _) = par_process_temp_buffer_to_mem_address_stage(
        UDynVarSys::from_n(2u8, 2),
        UDynVarSys::from_n(3u8, 2),
        &mut mach_input,
        temp_buffer_step,
        temp_buffer_step_pos,
        func,
    );
    // stop stage
    let mut output_4 = InfParOutputSys::new(config);
    output_4.state = mach_input.state.clone();
    output_4.stop = true.into();
    let mut output_stages = vec![output_1, output_2, output_3, output_4];
    InfParOutputSys::fix_state_len(&mut output_stages);
    let final_state = dynint_table(
        mach_input.state.clone().subvalue(0, 2),
        output_stages.into_iter().map(|v| v.to_dynintvar()),
    );
    mobj.in_state = Some(mach_input.state);
    mobj.from_dynintvar(final_state);
    mobj.to_machine().to_toml()
}

fn gen_process_mem_address_to_temp_buffer_test(
    cell_len_bits: u32,
    data_part_len: u32,
    temp_buffer_len: u32,
    proc_num: u64,
    mem_size: u64,
    temp_buffer_step: u32,
    temp_buffer_step_pos: u32,
    func: impl Function1,
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
    // first stage
    let (output_1, _) = init_machine_end_pos_stage(
        UDynVarSys::from_n(0u8, 2),
        UDynVarSys::from_n(1u8, 2),
        &mut mach_input,
        temp_buffer_step,
    );
    let (output_2, _) = par_copy_proc_id_to_mem_address_stage(
        UDynVarSys::from_n(1u8, 2),
        UDynVarSys::from_n(2u8, 2),
        &mut mach_input,
        temp_buffer_step,
    );
    let (output_3, _, _, _) = par_process_mem_address_to_temp_buffer_stage(
        UDynVarSys::from_n(2u8, 2),
        UDynVarSys::from_n(3u8, 2),
        &mut mach_input,
        temp_buffer_step,
        temp_buffer_step_pos,
        func,
    );
    // stop stage
    let mut output_4 = InfParOutputSys::new(config);
    output_4.state = mach_input.state.clone();
    output_4.stop = true.into();
    let mut output_stages = vec![output_1, output_2, output_3, output_4];
    InfParOutputSys::fix_state_len(&mut output_stages);
    let final_state = dynint_table(
        mach_input.state.clone().subvalue(0, 2),
        output_stages.into_iter().map(|v| v.to_dynintvar()),
    );
    mobj.in_state = Some(mach_input.state);
    mobj.from_dynintvar(final_state);
    mobj.to_machine().to_toml()
}

fn process_3_temp_buffer_2_to_mem_address(
    cell_len_bits: u32,
    data_part_len: u32,
    temp_buffer_len: u32,
    proc_num: u64,
    mem_size: u64,
    temp_buffer_step: u32,
    tbs_pos_1: u32,
    tbs_pos_2: u32,
    func1: impl Function1,
    func2: impl Function1,
    func3: impl Function2,
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
    // first stage
    let (output_1, _) = init_machine_end_pos_stage(
        UDynVarSys::from_n(0u8, 3),
        UDynVarSys::from_n(1u8, 3),
        &mut mach_input,
        temp_buffer_step,
    );
    let (output_2, _, _, _) = par_process_proc_id_to_temp_buffer_stage(
        UDynVarSys::from_n(1u8, 3),
        UDynVarSys::from_n(2u8, 3),
        &mut mach_input,
        temp_buffer_step,
        tbs_pos_1,
        func1,
    );
    let (output_3, _, _, _) = par_process_proc_id_to_temp_buffer_stage(
        UDynVarSys::from_n(2u8, 3),
        UDynVarSys::from_n(3u8, 3),
        &mut mach_input,
        temp_buffer_step,
        tbs_pos_2,
        func2,
    );
    let (output_4, _, _, _) = par_process_temp_buffer_2_to_mem_address_stage(
        UDynVarSys::from_n(3u8, 3),
        UDynVarSys::from_n(4u8, 3),
        &mut mach_input,
        temp_buffer_step,
        tbs_pos_1,
        tbs_pos_2,
        true,
        true,
        func3,
    );
    // stop stage
    let mut output_5 = InfParOutputSys::new(config);
    output_5.state = mach_input.state.clone();
    output_5.stop = true.into();
    let mut output_stages = vec![output_1, output_2, output_3, output_4, output_5.clone()];
    InfParOutputSys::fix_state_len(&mut output_stages);
    let final_state = dynint_table_partial(
        mach_input.state.clone().subvalue(0, 3),
        output_stages.into_iter().map(|v| v.to_dynintvar()),
        output_5.to_dynintvar(),
    );
    mobj.in_state = Some(mach_input.state);
    mobj.from_dynintvar(final_state);
    mobj.to_machine().to_toml()
}

fn gen_process_to_mem_address_test(
    cell_len_bits: u32,
    data_part_len: u32,
    temp_buffer_len: u32,
    proc_num: u64,
    mem_size: u64,
    temp_buffer_step: u32,
    func: impl Function0,
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
    // first stage
    let (output_1, _) = init_machine_end_pos_stage(
        UDynVarSys::from_n(0u8, 2),
        UDynVarSys::from_n(1u8, 2),
        &mut mach_input,
        temp_buffer_step,
    );
    let (output_2, _) = par_copy_proc_id_to_mem_address_stage(
        UDynVarSys::from_n(1u8, 2),
        UDynVarSys::from_n(2u8, 2),
        &mut mach_input,
        temp_buffer_step,
    );
    let (output_3, _, _, _) = par_process_to_mem_address_stage(
        UDynVarSys::from_n(2u8, 2),
        UDynVarSys::from_n(3u8, 2),
        &mut mach_input,
        temp_buffer_step,
        func,
    );
    // stop stage
    let mut output_4 = InfParOutputSys::new(config);
    output_4.state = mach_input.state.clone();
    output_4.stop = true.into();
    let mut output_stages = vec![output_1, output_2, output_3, output_4];
    InfParOutputSys::fix_state_len(&mut output_stages);
    let final_state = dynint_table(
        mach_input.state.clone().subvalue(0, 2),
        output_stages.into_iter().map(|v| v.to_dynintvar()),
    );
    mobj.in_state = Some(mach_input.state);
    mobj.from_dynintvar(final_state);
    mobj.to_machine().to_toml()
}

fn gen_process_to_temp_buffer_test(
    cell_len_bits: u32,
    data_part_len: u32,
    temp_buffer_len: u32,
    proc_num: u64,
    mem_size: u64,
    temp_buffer_step: u32,
    tbs_pos: u32,
    func: impl Function0,
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
    // first stage
    let (output_1, _) = init_machine_end_pos_stage(
        UDynVarSys::from_n(0u8, 2),
        UDynVarSys::from_n(1u8, 2),
        &mut mach_input,
        temp_buffer_step,
    );
    let (output_2, _) = par_copy_proc_id_to_temp_buffer_stage(
        UDynVarSys::from_n(1u8, 2),
        UDynVarSys::from_n(2u8, 2),
        &mut mach_input,
        temp_buffer_step,
        tbs_pos,
    );
    let (output_3, _, _, _) = par_process_to_temp_buffer_stage(
        UDynVarSys::from_n(2u8, 2),
        UDynVarSys::from_n(3u8, 2),
        &mut mach_input,
        temp_buffer_step,
        tbs_pos,
        false,
        func,
    );
    // stop stage
    let mut output_4 = InfParOutputSys::new(config);
    output_4.state = mach_input.state.clone();
    output_4.stop = true.into();
    let mut output_stages = vec![output_1, output_2, output_3, output_4];
    InfParOutputSys::fix_state_len(&mut output_stages);
    let final_state = dynint_table(
        mach_input.state.clone().subvalue(0, 2),
        output_stages.into_iter().map(|v| v.to_dynintvar()),
    );
    mobj.in_state = Some(mach_input.state);
    mobj.from_dynintvar(final_state);
    mobj.to_machine().to_toml()
}

//

fn parse_infdataparam_elem(s: &str) -> Result<(InfDataParam, u32), String> {
    // InfDataParam formats: "m" - mem_address, "p" - proc_id, "t4" - temp buffer pos 4,
    // "e7" - end pos index 7.
    // elem format: {InfDataParam}:end_pos. example: "t7:1"
    if s.is_empty() {
        return Err("Empty string".to_string());
    }
    let (param, r) = if s.starts_with("m") {
        (InfDataParam::MemAddress, &s[1..])
    } else if s.starts_with("p") {
        (InfDataParam::ProcId, &s[1..])
    } else if s.starts_with("t") || s.starts_with("e") {
        let is_end_pos = s.starts_with("e");
        let r = &s[1..];
        if let Some(end) = r.find(':') {
            let pos = r[0..end]
                .parse()
                .map_err(|e: std::num::ParseIntError| e.to_string())?;
            let param = if is_end_pos {
                InfDataParam::EndPos(pos)
            } else {
                InfDataParam::TempBuffer(pos)
            };
            (param, &r[end..])
        } else {
            return Err("No delimiter".to_string());
        }
    } else {
        return Err("Unknown type".to_string());
    };
    if r.starts_with(":") {
        let end_pos = r[1..]
            .parse()
            .map_err(|e: std::num::ParseIntError| e.to_string())?;
        Ok((param, end_pos))
    } else {
        Err("No delimiter".to_string())
    }
}

fn gen_process_infinite_data_test(
    cell_len_bits: u32,
    data_part_len: u32,
    temp_buffer_len: u32,
    proc_num: u64,
    mem_size: u64,
    temp_buffer_step: u32,
    src_params: &[(InfDataParam, u32)],
    dests: &[(InfDataParam, u32)],
    func: impl FunctionNN,
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
    // first stage
    let (output_1, _) = init_machine_end_pos_stage(
        UDynVarSys::from_n(0u8, 2),
        UDynVarSys::from_n(1u8, 2),
        &mut mach_input,
        temp_buffer_step,
    );
    let (output_2, _, _, _) = par_process_infinite_data_stage(
        UDynVarSys::from_n(1u8, 2),
        UDynVarSys::from_n(2u8, 2),
        &mut mach_input,
        temp_buffer_step,
        src_params,
        dests,
        func,
    );
    // stop stage
    let mut output_3 = InfParOutputSys::new(config);
    output_3.state = mach_input.state.clone();
    output_3.stop = true.into();
    let mut output_stages = vec![output_1, output_2, output_3.clone(), output_3];
    InfParOutputSys::fix_state_len(&mut output_stages);
    let final_state = dynint_table(
        mach_input.state.clone().subvalue(0, 2),
        output_stages.into_iter().map(|v| v.to_dynintvar()),
    );
    mobj.in_state = Some(mach_input.state);
    mobj.from_dynintvar(final_state);
    mobj.to_machine().to_toml()
}

fn gen_mem_data_to_start_test(
    cell_len_bits: u32,
    data_part_len: u32,
    temp_buffer_len: u32,
    proc_num: u64,
    mem_size: u64,
    temp_buffer_step: u32,
    proc_elem_len_bits: u32,
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
    // first stage
    let (output_1, _) = init_machine_end_pos_stage(
        UDynVarSys::from_n(0u8, 2),
        UDynVarSys::from_n(1u8, 2),
        &mut mach_input,
        temp_buffer_step,
    );
    // first stage
    let (output_2, _) = mem_data_to_start(
        UDynVarSys::from_n(1u8, 2),
        UDynVarSys::from_n(2u8, 2),
        &mut mach_input,
        temp_buffer_step,
        proc_elem_len_bits,
    );
    // stop stage
    let mut output_3 = InfParOutputSys::new(config);
    output_3.state = mach_input.state.clone();
    output_3.stop = true.into();
    let mut output_stages = vec![output_1, output_2, output_3.clone(), output_3];
    InfParOutputSys::fix_state_len(&mut output_stages);
    let final_state = dynint_table(
        mach_input.state.clone().subvalue(0, 2),
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
        "move_data_pos_expr" => {
            let step_num: u64 = args.next().unwrap().parse().unwrap();
            assert_ne!(step_num, 0);
            print!(
                "{}",
                callsys(|| gen_move_data_pos_expr_test(
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
        "inc_mem_address" => {
            let step_num: u64 = args.next().unwrap().parse().unwrap();
            assert_ne!(step_num, 0);
            print!(
                "{}",
                callsys(|| gen_inc_mem_address_test(
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
        "init_machine_end_pos_one" => {
            let temp_buffer_step: u32 = args.next().unwrap().parse().unwrap();
            assert_ne!(temp_buffer_step, 0);
            print!(
                "{}",
                callsys(|| gen_init_machine_end_pos_one_test(
                    cell_len_bits,
                    data_part_len,
                    temp_buffer_len,
                    proc_num,
                    mem_size,
                    temp_buffer_step,
                )
                .unwrap())
            );
        }
        "copy_proc_id_to_temp_buffer" => {
            let temp_buffer_step: u32 = args.next().unwrap().parse().unwrap();
            let temp_buffer_step_pos: u32 = args.next().unwrap().parse().unwrap();
            assert_ne!(temp_buffer_step, 0);
            print!(
                "{}",
                callsys(|| gen_copy_proc_id_to_temp_buffer_test(
                    cell_len_bits,
                    data_part_len,
                    temp_buffer_len,
                    proc_num,
                    mem_size,
                    temp_buffer_step,
                    temp_buffer_step_pos,
                )
                .unwrap())
            );
        }
        "copy_proc_id_to_mem_address" => {
            let temp_buffer_step: u32 = args.next().unwrap().parse().unwrap();
            assert_ne!(temp_buffer_step, 0);
            print!(
                "{}",
                callsys(|| gen_copy_proc_id_to_mem_address_test(
                    cell_len_bits,
                    data_part_len,
                    temp_buffer_len,
                    proc_num,
                    mem_size,
                    temp_buffer_step,
                )
                .unwrap())
            );
        }
        "copy_proc_id_to_mem_address_2" => {
            let temp_buffer_step: u32 = args.next().unwrap().parse().unwrap();
            assert_ne!(temp_buffer_step, 0);
            print!(
                "{}",
                callsys(|| gen_copy_proc_id_to_mem_address_test_2(
                    cell_len_bits,
                    data_part_len,
                    temp_buffer_len,
                    proc_num,
                    mem_size,
                    temp_buffer_step,
                )
                .unwrap())
            );
        }
        "copy_temp_buffer_to_mem_address" => {
            let temp_buffer_step: u32 = args.next().unwrap().parse().unwrap();
            let temp_buffer_step_pos: u32 = args.next().unwrap().parse().unwrap();
            assert_ne!(temp_buffer_step, 0);
            print!(
                "{}",
                callsys(|| gen_copy_temp_buffer_to_mem_address_test(
                    cell_len_bits,
                    data_part_len,
                    temp_buffer_len,
                    proc_num,
                    mem_size,
                    temp_buffer_step,
                    temp_buffer_step_pos,
                )
                .unwrap())
            );
        }
        "copy_mem_address_to_temp_buffer" => {
            let temp_buffer_step: u32 = args.next().unwrap().parse().unwrap();
            let temp_buffer_step_pos: u32 = args.next().unwrap().parse().unwrap();
            assert_ne!(temp_buffer_step, 0);
            print!(
                "{}",
                callsys(|| gen_copy_mem_address_to_temp_buffer_test(
                    cell_len_bits,
                    data_part_len,
                    temp_buffer_len,
                    proc_num,
                    mem_size,
                    temp_buffer_step,
                    temp_buffer_step_pos,
                )
                .unwrap())
            );
        }
        "copy_temp_buffer_to_temp_buffer" => {
            let temp_buffer_step: u32 = args.next().unwrap().parse().unwrap();
            let tbs_src_pos: u32 = args.next().unwrap().parse().unwrap();
            let tbs_dest_pos: u32 = args.next().unwrap().parse().unwrap();
            let proc_id_end_pos: u32 = args.next().unwrap().parse().unwrap();
            assert_ne!(temp_buffer_step, 0);
            print!(
                "{}",
                callsys(|| gen_copy_temp_buffer_to_temp_buffer_test(
                    cell_len_bits,
                    data_part_len,
                    temp_buffer_len,
                    proc_num,
                    mem_size,
                    temp_buffer_step,
                    tbs_src_pos,
                    tbs_dest_pos,
                    proc_id_end_pos != 0,
                )
                .unwrap())
            );
        }
        "add_proc_id_to_temp_buffer" => {
            let temp_buffer_step: u32 = args.next().unwrap().parse().unwrap();
            let temp_buffer_step_pos: u32 = args.next().unwrap().parse().unwrap();
            let value: u64 = args.next().unwrap().parse().unwrap();
            assert_ne!(temp_buffer_step, 0);
            print!(
                "{}",
                callsys(|| gen_process_proc_id_to_temp_buffer_test(
                    cell_len_bits,
                    data_part_len,
                    temp_buffer_len,
                    proc_num,
                    mem_size,
                    temp_buffer_step,
                    temp_buffer_step_pos,
                    Add1Func::new_from_u64(data_part_len as usize, value),
                )
                .unwrap())
            );
        }
        "add_proc_id_to_mem_address" => {
            let temp_buffer_step: u32 = args.next().unwrap().parse().unwrap();
            let value: u64 = args.next().unwrap().parse().unwrap();
            assert_ne!(temp_buffer_step, 0);
            print!(
                "{}",
                callsys(|| gen_process_proc_id_to_mem_address_test(
                    cell_len_bits,
                    data_part_len,
                    temp_buffer_len,
                    proc_num,
                    mem_size,
                    temp_buffer_step,
                    Add1Func::new_from_u64(data_part_len as usize, value),
                )
                .unwrap())
            );
        }
        "add_signed_proc_id_to_mem_address" => {
            let temp_buffer_step: u32 = args.next().unwrap().parse().unwrap();
            let value: i64 = args.next().unwrap().parse().unwrap();
            assert_ne!(temp_buffer_step, 0);
            print!(
                "{}",
                callsys(|| gen_process_proc_id_to_mem_address_test(
                    cell_len_bits,
                    data_part_len,
                    temp_buffer_len,
                    proc_num,
                    mem_size,
                    temp_buffer_step,
                    Add1Func::new_from_i64(data_part_len as usize, value),
                )
                .unwrap())
            );
        }
        "sub_proc_id_to_mem_address" => {
            let temp_buffer_step: u32 = args.next().unwrap().parse().unwrap();
            let value: u64 = args.next().unwrap().parse().unwrap();
            assert_ne!(temp_buffer_step, 0);
            print!(
                "{}",
                callsys(|| gen_process_proc_id_to_mem_address_test(
                    cell_len_bits,
                    data_part_len,
                    temp_buffer_len,
                    proc_num,
                    mem_size,
                    temp_buffer_step,
                    Sub1Func::new_from_u64(data_part_len as usize, value),
                )
                .unwrap())
            );
        }
        "sub_signed_proc_id_to_mem_address" => {
            let temp_buffer_step: u32 = args.next().unwrap().parse().unwrap();
            let value: i64 = args.next().unwrap().parse().unwrap();
            assert_ne!(temp_buffer_step, 0);
            print!(
                "{}",
                callsys(|| gen_process_proc_id_to_mem_address_test(
                    cell_len_bits,
                    data_part_len,
                    temp_buffer_len,
                    proc_num,
                    mem_size,
                    temp_buffer_step,
                    Sub1Func::new_from_i64(data_part_len as usize, value),
                )
                .unwrap())
            );
        }
        "mul_proc_id_to_mem_address" => {
            let temp_buffer_step: u32 = args.next().unwrap().parse().unwrap();
            let value: u64 = args.next().unwrap().parse().unwrap();
            assert_ne!(temp_buffer_step, 0);
            print!(
                "{}",
                callsys(|| gen_process_proc_id_to_mem_address_test(
                    cell_len_bits,
                    data_part_len,
                    temp_buffer_len,
                    proc_num,
                    mem_size,
                    temp_buffer_step,
                    Mul1Func::new_from_u64(data_part_len as usize, value),
                )
                .unwrap())
            );
        }
        "muladd_proc_id_to_mem_address" => {
            let temp_buffer_step: u32 = args.next().unwrap().parse().unwrap();
            let value1: u64 = args.next().unwrap().parse().unwrap();
            let value2: u64 = args.next().unwrap().parse().unwrap();
            assert_ne!(temp_buffer_step, 0);
            print!(
                "{}",
                callsys(|| gen_process_proc_id_to_mem_address_test(
                    cell_len_bits,
                    data_part_len,
                    temp_buffer_len,
                    proc_num,
                    mem_size,
                    temp_buffer_step,
                    MulAdd1Func::new_from_u64(data_part_len as usize, value1, value2),
                )
                .unwrap())
            );
        }
        "align_proc_id_to_mem_address" => {
            let temp_buffer_step: u32 = args.next().unwrap().parse().unwrap();
            let bits: u64 = args.next().unwrap().parse().unwrap();
            assert_ne!(temp_buffer_step, 0);
            print!(
                "{}",
                callsys(|| gen_process_proc_id_to_mem_address_test(
                    cell_len_bits,
                    data_part_len,
                    temp_buffer_len,
                    proc_num,
                    mem_size,
                    temp_buffer_step,
                    Align1Func::new(data_part_len as usize, bits),
                )
                .unwrap())
            );
        }
        "add_temp_buffer_to_mem_address" => {
            let temp_buffer_step: u32 = args.next().unwrap().parse().unwrap();
            let temp_buffer_step_pos: u32 = args.next().unwrap().parse().unwrap();
            let value: u64 = args.next().unwrap().parse().unwrap();
            assert_ne!(temp_buffer_step, 0);
            print!(
                "{}",
                callsys(|| gen_process_temp_buffer_to_mem_address_test(
                    cell_len_bits,
                    data_part_len,
                    temp_buffer_len,
                    proc_num,
                    mem_size,
                    temp_buffer_step,
                    temp_buffer_step_pos,
                    Add1Func::new_from_u64(data_part_len as usize, value),
                )
                .unwrap())
            );
        }
        "add_mem_address_to_temp_buffer" => {
            let temp_buffer_step: u32 = args.next().unwrap().parse().unwrap();
            let temp_buffer_step_pos: u32 = args.next().unwrap().parse().unwrap();
            let value: u64 = args.next().unwrap().parse().unwrap();
            assert_ne!(temp_buffer_step, 0);
            print!(
                "{}",
                callsys(|| gen_process_mem_address_to_temp_buffer_test(
                    cell_len_bits,
                    data_part_len,
                    temp_buffer_len,
                    proc_num,
                    mem_size,
                    temp_buffer_step,
                    temp_buffer_step_pos,
                    Add1Func::new_from_u64(data_part_len as usize, value),
                )
                .unwrap())
            );
        }
        "shl_proc_id_to_mem_address" => {
            let temp_buffer_step: u32 = args.next().unwrap().parse().unwrap();
            let shift: usize = args.next().unwrap().parse().unwrap();
            assert_ne!(temp_buffer_step, 0);
            print!(
                "{}",
                callsys(|| gen_process_proc_id_to_mem_address_test(
                    cell_len_bits,
                    data_part_len,
                    temp_buffer_len,
                    proc_num,
                    mem_size,
                    temp_buffer_step,
                    Shl1Func::new(data_part_len as usize, shift),
                )
                .unwrap())
            );
        }
        "xornn_process_infinite_data_test" => {
            // dummy test for testing par_process_infinite_data_stage
            let temp_buffer_step: u32 = args.next().unwrap().parse().unwrap();
            let src_params_str = args.next().unwrap();
            let src_params = if !src_params_str.is_empty() {
                src_params_str
                    .split(',')
                    .map(|x| parse_infdataparam_elem(x).unwrap())
                    .collect::<Vec<_>>()
            } else {
                vec![]
            };
            let dests = args
                .next()
                .unwrap()
                .split(',')
                .map(|x| parse_infdataparam_elem(x).unwrap())
                .collect::<Vec<_>>();
            assert_ne!(temp_buffer_step, 0);
            // println!("TempBuferStep: {}", temp_buffer_step);
            // println!("SrcParams: {:?}", src_params);
            // println!("Dests: {:?}", dests);
            print!(
                "{}",
                callsys(|| gen_process_infinite_data_test(
                    cell_len_bits,
                    data_part_len,
                    temp_buffer_len,
                    proc_num,
                    mem_size,
                    temp_buffer_step,
                    &src_params,
                    &dests,
                    XorNNFuncSample::new(data_part_len as usize, src_params.len(), dests.len()),
                )
                .unwrap())
            );
        }
        "xor_add_temp_buffer_2_to_mem_address" => {
            let temp_buffer_step: u32 = args.next().unwrap().parse().unwrap();
            let tbs_pos_1: u32 = args.next().unwrap().parse().unwrap();
            let tbs_pos_2: u32 = args.next().unwrap().parse().unwrap();
            let value1: u64 = args.next().unwrap().parse().unwrap();
            let value2: u64 = args.next().unwrap().parse().unwrap();
            assert_ne!(temp_buffer_step, 0);
            print!(
                "{}",
                callsys(|| process_3_temp_buffer_2_to_mem_address(
                    cell_len_bits,
                    data_part_len,
                    temp_buffer_len,
                    proc_num,
                    mem_size,
                    temp_buffer_step,
                    tbs_pos_1,
                    tbs_pos_2,
                    Xor1Func::new_from_u64(data_part_len as usize, value1),
                    Xor1Func::new_from_u64(data_part_len as usize, value2),
                    Add2Func::new(),
                )
                .unwrap())
            );
        }
        "xor_sub_temp_buffer_2_to_mem_address" => {
            let temp_buffer_step: u32 = args.next().unwrap().parse().unwrap();
            let tbs_pos_1: u32 = args.next().unwrap().parse().unwrap();
            let tbs_pos_2: u32 = args.next().unwrap().parse().unwrap();
            let value1: u64 = args.next().unwrap().parse().unwrap();
            let value2: u64 = args.next().unwrap().parse().unwrap();
            assert_ne!(temp_buffer_step, 0);
            print!(
                "{}",
                callsys(|| process_3_temp_buffer_2_to_mem_address(
                    cell_len_bits,
                    data_part_len,
                    temp_buffer_len,
                    proc_num,
                    mem_size,
                    temp_buffer_step,
                    tbs_pos_1,
                    tbs_pos_2,
                    Xor1Func::new_from_u64(data_part_len as usize, value1),
                    Xor1Func::new_from_u64(data_part_len as usize, value2),
                    Sub2Func::new(),
                )
                .unwrap())
            );
        }
        "mem_data_to_start" => {
            let temp_buffer_step: u32 = args.next().unwrap().parse().unwrap();
            let proc_elem_len_bits: u32 = args.next().unwrap().parse().unwrap();
            assert_ne!(temp_buffer_step, 0);
            print!(
                "{}",
                callsys(|| gen_mem_data_to_start_test(
                    cell_len_bits,
                    data_part_len,
                    temp_buffer_len,
                    proc_num,
                    mem_size,
                    temp_buffer_step,
                    proc_elem_len_bits,
                )
                .unwrap())
            );
        }
        "add_proc_id_to_mem_address_ext_out" => {
            let temp_buffer_step: u32 = args.next().unwrap().parse().unwrap();
            let value: u64 = args.next().unwrap().parse().unwrap();
            assert_ne!(temp_buffer_step, 0);
            print!(
                "{}",
                callsys(|| gen_process_proc_id_to_mem_address_ext_out_test(
                    cell_len_bits,
                    data_part_len,
                    temp_buffer_len,
                    proc_num,
                    mem_size,
                    temp_buffer_step,
                    0,
                    0,
                    Add1Func::new_from_u64(data_part_len as usize, value),
                )
                .unwrap())
            );
        }
        "sub_proc_id_to_mem_address_ext_out" => {
            let temp_buffer_step: u32 = args.next().unwrap().parse().unwrap();
            let value: u64 = args.next().unwrap().parse().unwrap();
            assert_ne!(temp_buffer_step, 0);
            print!(
                "{}",
                callsys(|| gen_process_proc_id_to_mem_address_ext_out_test(
                    cell_len_bits,
                    data_part_len,
                    temp_buffer_len,
                    proc_num,
                    mem_size,
                    temp_buffer_step,
                    0,
                    0,
                    Sub1Func::new_from_u64(data_part_len as usize, value),
                )
                .unwrap())
            );
        }
        "zero_to_mem_address" => {
            let temp_buffer_step: u32 = args.next().unwrap().parse().unwrap();
            assert_ne!(temp_buffer_step, 0);
            print!(
                "{}",
                callsys(|| gen_process_to_mem_address_test(
                    cell_len_bits,
                    data_part_len,
                    temp_buffer_len,
                    proc_num,
                    mem_size,
                    temp_buffer_step,
                    Zero0Func::new(data_part_len as usize)
                )
                .unwrap())
            );
        }
        "one_to_mem_address" => {
            let temp_buffer_step: u32 = args.next().unwrap().parse().unwrap();
            assert_ne!(temp_buffer_step, 0);
            print!(
                "{}",
                callsys(|| gen_process_to_mem_address_test(
                    cell_len_bits,
                    data_part_len,
                    temp_buffer_len,
                    proc_num,
                    mem_size,
                    temp_buffer_step,
                    One0Func::new(data_part_len as usize)
                )
                .unwrap())
            );
        }
        "zero_to_temp_buffer" => {
            let temp_buffer_step: u32 = args.next().unwrap().parse().unwrap();
            let tbs_pos: u32 = args.next().unwrap().parse().unwrap();
            assert_ne!(temp_buffer_step, 0);
            print!(
                "{}",
                callsys(|| gen_process_to_temp_buffer_test(
                    cell_len_bits,
                    data_part_len,
                    temp_buffer_len,
                    proc_num,
                    mem_size,
                    temp_buffer_step,
                    tbs_pos,
                    Zero0Func::new(data_part_len as usize)
                )
                .unwrap())
            );
        }
        "one_to_temp_buffer" => {
            let temp_buffer_step: u32 = args.next().unwrap().parse().unwrap();
            let tbs_pos: u32 = args.next().unwrap().parse().unwrap();
            assert_ne!(temp_buffer_step, 0);
            print!(
                "{}",
                callsys(|| gen_process_to_temp_buffer_test(
                    cell_len_bits,
                    data_part_len,
                    temp_buffer_len,
                    proc_num,
                    mem_size,
                    temp_buffer_step,
                    tbs_pos,
                    One0Func::new(data_part_len as usize)
                )
                .unwrap())
            );
        }
        _ => {
            panic!("Unknown example");
        }
    }
}
