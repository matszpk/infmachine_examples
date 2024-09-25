use super::*;

// move to endpos

pub fn par_move_to_end_pos_stage(
    output_state: UDynVarSys,
    next_state: UDynVarSys,
    input: &mut InfParInputSys,
    temp_buffer_step: u32,
    end_pos: u32,
    mem_address: bool,
    proc_id: bool,
) -> (InfParOutputSys, BoolVarSys) {
    assert_ne!(temp_buffer_step, 0);
    let config = input.config();
    let dp_len = config.data_part_len;
    assert!(end_pos < temp_buffer_step * dp_len);
    let total_stages =
        2 * usize::from(end_pos >= dp_len) + 3 + usize::from(mem_address) + usize::from(proc_id);
    let state_start = output_state.bitnum();
    let stage_type_len = calc_log_bits(total_stages);
    extend_output_state(state_start, stage_type_len, input);
    let stage = input.state.clone().subvalue(state_start, stage_type_len);
    let create_out_state = |s| output_state.clone().concat(s);
    let output_base = InfParOutputSys::new(config);
    // move to end_pos
    let mut outputs = vec![];
    if end_pos >= dp_len {
        // move to end pos
        let (output, _) = move_data_pos_stage(
            create_out_state(UDynVarSys::from_n(outputs.len(), stage_type_len)),
            create_out_state(UDynVarSys::from_n(outputs.len() + 1, stage_type_len)),
            input,
            DKIND_TEMP_BUFFER,
            DPMOVE_FORWARD,
            (end_pos / dp_len) as u64,
        );
        outputs.push(output);
    }
    let loop_start = outputs.len();
    // read temp buffer end pos
    let mut output = output_base.clone();
    output.state = create_out_state(UDynVarSys::from_n(outputs.len() + 1, stage_type_len));
    output.dkind = DKIND_TEMP_BUFFER.into();
    output.dpr = true.into();
    outputs.push(output);
    // check if not zero
    let end_stage = if end_pos >= dp_len {
        outputs.len() + 2 + usize::from(mem_address) + usize::from(proc_id)
    } else {
        outputs.len()
    };
    let mut end_of_stage = input.dpval.bit((end_pos % dp_len) as usize);
    let mut output = output_base.clone();
    output.state = create_out_state(dynint_ite(
        end_of_stage.clone(),
        UDynVarSys::from_n(end_stage, stage_type_len),
        UDynVarSys::from_n(outputs.len() + 1, stage_type_len),
    ));
    outputs.push(output);
    // move temp buffer
    let (output, _) = move_data_pos_stage(
        create_out_state(UDynVarSys::from_n(outputs.len(), stage_type_len)),
        create_out_state(if mem_address || proc_id {
            UDynVarSys::from_n(outputs.len() + 1, stage_type_len)
        } else {
            UDynVarSys::from_n(loop_start, stage_type_len)
        }),
        input,
        DKIND_TEMP_BUFFER,
        DPMOVE_FORWARD,
        temp_buffer_step as u64,
    );
    outputs.push(output);
    // if set move mem_address
    if mem_address {
        let mut output = output_base.clone();
        output.state = create_out_state(if proc_id {
            UDynVarSys::from_n(outputs.len() + 1, stage_type_len)
        } else {
            UDynVarSys::from_n(loop_start, stage_type_len)
        });
        output.dkind = DKIND_MEM_ADDRESS.into();
        output.dpmove = DPMOVE_FORWARD.into();
        outputs.push(output);
    }
    if proc_id {
        let mut output = output_base.clone();
        output.state = create_out_state(UDynVarSys::from_n(loop_start, stage_type_len));
        output.dkind = DKIND_PROC_ID.into();
        output.dpmove = DPMOVE_FORWARD.into();
        outputs.push(output);
    }
    if end_pos >= dp_len {
        assert_eq!(end_stage, outputs.len());
        // end: move to start of chunk in temp buffer
        let (output, end) = move_data_pos_stage(
            create_out_state(UDynVarSys::from_n(outputs.len(), stage_type_len)),
            create_out_state(UDynVarSys::from_n(0u8, stage_type_len)),
            input,
            DKIND_TEMP_BUFFER,
            DPMOVE_BACKWARD,
            (end_pos / dp_len) as u64,
        );
        end_of_stage = end;
        outputs.push(output);
    }
    assert_eq!(total_stages, outputs.len());
    // prepare end bit
    let end = (&stage).equal(end_stage) & end_of_stage;
    // finish generation
    finish_stage_with_table(output_state, next_state, input, outputs, stage, end)
}

// macro_rules! test_println {
//     () => { eprintln!(); };
//     ($($arg:tt)*) => { eprintln!($($arg)*); };
// }

macro_rules! test_println {
    () => {};
    ($($arg:tt)*) => {};
}

// main routine to process infinite data (mem_address, proc_id and temp_buffer).
// return (parmachine_output, end variable, external_outputs, external_outputs set variable)
pub fn par_process_infinite_data_stage<F: FunctionNN>(
    output_state: UDynVarSys,
    next_state: UDynVarSys,
    input: &mut InfParInputSys,
    temp_buffer_step: u32,
    src_params: &[(InfDataParam, u32)],
    dests: &[(InfDataParam, u32)],
    func: F,
) -> (InfParOutputSys, BoolVarSys, Vec<UDynVarSys>, BoolVarSys) {
    let src_len = src_params.len();
    let dest_len = dests.len();
    assert_ne!(temp_buffer_step, 0);
    assert_eq!(output_state.bitnum(), next_state.bitnum());
    assert_eq!(func.input_num(), src_len);
    assert_eq!(func.output_num(), dest_len);
    let config = input.config();
    let dp_len = config.data_part_len;
    // src_params can be empty (no input for functions)
    assert!(!dests.is_empty());
    for (data_param, end_pos) in src_params.iter().chain(dests.iter()) {
        let good = match data_param {
            InfDataParam::TempBuffer(pos) => *pos < temp_buffer_step,
            InfDataParam::EndPos(idx) => *idx < dp_len * temp_buffer_step,
            _ => true,
        };
        assert!(good && *end_pos < dp_len * temp_buffer_step);
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
            "  EndPos: TotalStages: {}, TotalStateBits: {}, LastPos: {}",
            total_stages,
            total_state_bits,
            last_pos
        );
    }
    test_println!("  ReadPhase");
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
                read_state_bits += dp_len as usize;
            }
            _ => {
                read_state_bits += dp_len as usize;
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
                write_state_bits += dp_len as usize;
            }
            _ => {
                write_state_bits += dp_len as usize;
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
        "  End: TotalStages: {}, TotalStateBits: {}, LastPos: {}",
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
                last_pos = pos;
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
                    "    GenEndPos {} {} {}: Read stage: {}",
                    i,
                    end_pos,
                    outputs.len(),
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
                            state_vars.bit(start + i + x)
                                | input.dpval.bit((end_pos % dp_len) as usize)
                        }),
                );
                test_println!(
                    "    GenEndPos {} {} {}: Write stage: {}: {:?}",
                    i,
                    end_pos,
                    outputs.len(),
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
            dp_len as usize
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
                dp_len as usize
            };
            func_inputs.push(dynint_ite(
                // use src end_pos to filter function inputs (if 1 then zeroing)
                !state_vars.bit(i),
                UDynVarSys::from_iter((0..param_len).map(|x| state_vars.bit(state_pos + x))),
                UDynVarSys::from_n(0u8, param_len),
            ));
            test_println!(
                "  FuncInputs: {} {:?}: StatePos: {}, ParamLen: {}",
                i,
                param,
                state_pos,
                param_len
            );
            state_pos += param_len;
        }
        func_inputs
    };
    let (next_func_state, outvals, ext_outputs) = func.output(func_state.clone(), &func_inputs);
    let mut output = output_base.clone();
    // get function output bitvector
    let func_outputs = {
        let mut func_output_bits = vec![];
        for ((param, _), outval) in dests.into_iter().zip(outvals.into_iter()) {
            let param_len = if matches!(*param, InfDataParam::EndPos(_)) {
                1
            } else {
                dp_len as usize
            };
            func_output_bits.extend((0..param_len).map(|x| outval.bit(x)));
            test_println!("  FuncOutputs: {:?}: ParamLen: {}", param, param_len);
        }
        if read_state_bits > write_state_bits {
            // fix length of func output bits - fix if read state bits is longer
            // than write state bits
            func_output_bits
                .extend((write_state_bits..read_state_bits).map(|_| BoolVarSys::from(false)));
        }
        UDynVarSys::from_iter(func_output_bits)
    };
    // AND for all dest end_pos: E0 and E1 and E2 ... EN. If 1 then go to end.
    let end_of_process = (src_len..src_len + dest_len)
        .fold(BoolVarSys::from(true), |a, x| a.clone() & state_vars.bit(x));
    let next_stage = dynint_ite(
        end_of_process.clone(),
        UDynVarSys::from_n(end_stage, stage_type_len),
        UDynVarSys::from_n(outputs.len() + 1, stage_type_len),
    );
    test_println!(
        "  NextStage: {}..{}: EndStage: {}",
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
    let ext_outputs_set = !end_of_process & (&stage).equal(outputs.len());
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
                let bit = (p % dp_len) as usize;
                output.dpval = UDynVarSys::from_iter((0..dp_len as usize).map(|x| {
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
                output.dpval = UDynVarSys::from_iter(
                    (0..dp_len as usize).map(|x| state_vars.bit(state_pos + x)),
                );
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
                state_pos += dp_len as usize;
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
        (temp_buffer_step - last_pos) as u64,
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
    let (output, end) =
        finish_stage_with_table(output_state, next_state, input, outputs, stage, end);
    (output, end, ext_outputs, ext_outputs_set)
}

pub struct AlignShl2Func {
    align: Align1Func,
    shl: Shl1Func,
}

impl AlignShl2Func {
    pub fn new(inout_len: usize, bits: u32) -> Self {
        Self {
            align: Align1Func::new(inout_len, bits as u64),
            shl: Shl1Func::new(inout_len, bits as usize),
        }
    }
}

impl FunctionNN for AlignShl2Func {
    fn state_len(&self) -> usize {
        self.align.state_len() + self.shl.state_len()
    }
    fn input_num(&self) -> usize {
        2
    }
    fn output_num(&self) -> usize {
        2
    }
    // i0 - mem_address, i1 - proc_id
    // o0 - temp_buffer[first_pos], o1 - temp_buffer[second_pos]
    fn output(
        &self,
        input_state: UDynVarSys,
        input: &[UDynVarSys],
    ) -> (UDynVarSys, Vec<UDynVarSys>, Vec<UDynVarSys>) {
        let (align_state, shl_state) = input_state.split(self.align.state_len());
        let (align_next_state, align_result, align_ext_outputs) =
            self.align.output(align_state, input[0].clone());
        let (shl_next_state, shl_result, shl_ext_outputs) =
            self.shl.output(shl_state, input[1].clone());
        let mut ext_outputs = align_ext_outputs;
        ext_outputs.extend(shl_ext_outputs);
        (
            align_next_state.concat(shl_next_state),
            vec![align_result, shl_result],
            ext_outputs,
        )
    }
}

pub struct SwapAdd2Func {
    add: Add1Func,
}

impl SwapAdd2Func {
    pub fn new(inout_len: usize, value: UDynVarSys) -> Self {
        Self {
            add: Add1Func::new(inout_len, value),
        }
    }
    pub fn new_from_u64(inout_len: usize, value: u64) -> Self {
        Self {
            add: Add1Func::new_from_u64(inout_len, value),
        }
    }
}

impl FunctionNN for SwapAdd2Func {
    fn state_len(&self) -> usize {
        self.add.state_len()
    }
    fn input_num(&self) -> usize {
        1
    }
    fn output_num(&self) -> usize {
        2
    }
    // i0 - temp_buffer[second_pos]
    // o0 - mem_address, o1 - temp_buffer[second_pos]
    fn output(
        &self,
        input_state: UDynVarSys,
        input: &[UDynVarSys],
    ) -> (UDynVarSys, Vec<UDynVarSys>, Vec<UDynVarSys>) {
        let (add_next_state, add_result, add_ext_outputs) =
            self.add.output(input_state, input[0].clone());
        (
            add_next_state,
            vec![input[0].clone(), add_result],
            add_ext_outputs,
        )
    }
}

pub fn mem_data_to_start(
    output_state: UDynVarSys,
    next_state: UDynVarSys,
    input: &mut InfParInputSys,
    temp_buffer_step: u32,
    proc_elem_len_bits: u32,
) -> (InfParOutputSys, BoolVarSys) {
    assert_eq!(output_state.bitnum(), next_state.bitnum());
    assert_ne!(temp_buffer_step, 0);
    let proc_elem_len = 1u64 << proc_elem_len_bits;
    let config = input.config();
    let cell_len = 1 << config.cell_len_bits;
    let dp_len = config.data_part_len as usize;
    let state_start = output_state.bitnum();
    let index_bits = std::cmp::max(1, usize::try_from(proc_elem_len_bits).unwrap());
    type StageType = U3VarSys;
    extend_output_state(state_start, StageType::BITS + index_bits + cell_len, input);
    let stage =
        StageType::try_from(input.state.clone().subvalue(state_start, StageType::BITS)).unwrap();
    let index_count = input
        .state
        .clone()
        .subvalue(state_start + StageType::BITS, index_bits);
    let mem_value = input
        .state
        .clone()
        .subvalue(state_start + StageType::BITS + index_bits, cell_len);
    let output_base = InfParOutputSys::new(config);
    let create_out_state =
        |s: StageType, ic, mv| output_state.clone().concat(s.into()).concat(ic).concat(mv);

    let (first_pos, second_pos) = if dp_len == 1 { (2, 3) } else { (1, 2) };
    assert!(second_pos < temp_buffer_step);
    // Repeat loop by proc_len:
    // 1. temp_buffer[first_pos] = align_to_pow2(mem_address),
    //    temp_buffer[second_pos] = proc_id*proc_elem_len.
    let (output_0, _, _, _) = par_process_infinite_data_stage(
        create_out_state(StageType::from(0u8), index_count.clone(), mem_value.clone()),
        create_out_state(StageType::from(1u8), index_count.clone(), mem_value.clone()),
        input,
        temp_buffer_step,
        &[
            (InfDataParam::MemAddress, END_POS_MEM_ADDRESS),
            (InfDataParam::ProcId, END_POS_PROC_ID),
        ],
        &[
            (InfDataParam::TempBuffer(first_pos), END_POS_MEM_ADDRESS),
            (InfDataParam::TempBuffer(second_pos), END_POS_MEM_ADDRESS),
        ],
        AlignShl2Func::new(dp_len, proc_elem_len_bits),
    );
    // 2. mem_address = temp_buffer[first_pos] + temp_buffer[second_pos].
    let (output_1, _, _, _) = par_process_temp_buffer_2_to_mem_address_stage(
        create_out_state(StageType::from(1u8), index_count.clone(), mem_value.clone()),
        create_out_state(StageType::from(2u8), index_count.clone(), mem_value.clone()),
        input,
        temp_buffer_step,
        first_pos,
        second_pos,
        false,
        false,
        Add2Func::new(),
    );
    // 3. Read memory cell.
    let mut output_2 = output_base.clone();
    output_2.state = create_out_state(StageType::from(3u8), index_count.clone(), mem_value.clone());
    output_2.memr = true.into();
    // 4. store memory cell to state.
    let mut output_3 = output_base.clone();
    output_3.state = create_out_state(
        StageType::from(4u8),
        index_count.clone(),
        input.memval.clone(),
    );
    // 5. mem_address = temp_buffer[second_pos],
    //    temp_buffer[second_pos] = temp_buffer[second_pos] + 1.
    let (output_4, _, _, _) = par_process_infinite_data_stage(
        create_out_state(StageType::from(4u8), index_count.clone(), mem_value.clone()),
        create_out_state(StageType::from(5u8), index_count.clone(), mem_value.clone()),
        input,
        temp_buffer_step,
        &[(InfDataParam::TempBuffer(second_pos), END_POS_MEM_ADDRESS)],
        &[
            (InfDataParam::MemAddress, END_POS_MEM_ADDRESS),
            (InfDataParam::TempBuffer(second_pos), END_POS_MEM_ADDRESS),
        ],
        SwapAdd2Func::new_from_u64(dp_len, 1),
    );
    // 6. Write memory cell and store to state.
    // 7. If index != proc_elem_len-1 then index+=1 and go to 2 else end.
    let mut output_5 = output_base.clone();
    output_5.state = create_out_state(StageType::from(1u8), &index_count + 1u8, mem_value.clone());
    output_5.memw = true.into();
    output_5.memval = mem_value.clone();
    // prepare end bit
    let end = (&stage).equal(5u8) & (&index_count).equal(proc_elem_len - 1);
    let outputs = vec![output_0, output_1, output_2, output_3, output_4, output_5];
    // finish generation
    finish_stage_with_table(output_state, next_state, input, outputs, stage.into(), end)
}
