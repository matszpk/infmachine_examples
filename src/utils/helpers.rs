use std::io::Write;

use super::*;

pub const fn calc_log_bits(n: usize) -> usize {
    let nbits = usize::BITS - n.leading_zeros();
    if (1 << (nbits - 1)) == n {
        (nbits - 1) as usize
    } else {
        nbits as usize
    }
}

pub const fn calc_log_bits_u64(n: u64) -> usize {
    let nbits = u64::BITS - n.leading_zeros();
    if (1 << (nbits - 1)) == n {
        (nbits - 1) as usize
    } else {
        nbits as usize
    }
}

pub struct CellVec {
    cell_len_bits: u32,
    len: u64,
    data: Vec<u8>,
}

impl CellVec {
    pub fn new(cell_len_bits: u32) -> Self {
        Self {
            cell_len_bits,
            len: 0,
            data: vec![],
        }
    }

    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn push(&mut self, value: u64) {
        if self.cell_len_bits < 3 {
            let cell_mask = (1 << (1 << self.cell_len_bits)) - 1;
            let cell_addr_mask = (1 << (3 - self.cell_len_bits)) - 1;
            let cell_pos = self.len & cell_addr_mask;
            if cell_pos == 0 {
                self.data.push(0u8);
            }
            let shift = cell_pos << self.cell_len_bits;
            let vlen = usize::try_from(self.len >> (3 - self.cell_len_bits)).unwrap();
            self.data[vlen] |= (u8::try_from(value).unwrap() & cell_mask) << shift;
        } else {
            let val_bytes = value.to_le_bytes();
            let cell_len = 1 << (self.cell_len_bits - 3);
            let val_slice = &val_bytes[0..std::cmp::min(val_bytes.len(), cell_len)];
            self.data.extend(
                val_slice
                    .into_iter()
                    .copied()
                    .chain(std::iter::repeat(0u8).take(cell_len - val_slice.len())),
            );
        }
        self.len += 1;
    }
    pub fn to_vec(self) -> Vec<u8> {
        self.data
    }
}

pub struct CellWriter<W: Write> {
    cell_len_bits: u32,
    writer: Option<W>,
    first: bool,
    cell_pos: u32,
    cell: Vec<u8>,
}

impl<W: Write> CellWriter<W> {
    pub fn new(cell_len_bits: u32, writer: W) -> Self {
        let cell_byte_num = if cell_len_bits >= 3 {
            1 << (cell_len_bits - 3)
        } else {
            1
        };
        Self {
            cell_len_bits,
            writer: Some(writer),
            first: true,
            cell_pos: 0,
            cell: vec![0u8; cell_byte_num],
        }
    }

    pub fn write_cell(&mut self, cell: u64) -> std::io::Result<()> {
        if self.cell_len_bits < 3 {
            let cell_mask = (1 << (1 << self.cell_len_bits)) - 1;
            let cell_num_in_byte = 1 << (3 - self.cell_len_bits);
            let shift = self.cell_pos << self.cell_len_bits;
            self.cell[0] |= (u8::try_from(cell).unwrap() & cell_mask) << shift;
            self.cell_pos += 1;
            if self.cell_pos == cell_num_in_byte {
                self.cell_pos = 0;
            }
        } else {
            let cell_byte_num = 1 << (self.cell_len_bits - 3);
            let vbytes = cell.to_le_bytes();
            let byte_num = std::cmp::min(8, cell_byte_num);
            self.cell[0..byte_num].copy_from_slice(&vbytes[0..byte_num]);
        }
        self.first = false;
        if self.cell_pos == 0 {
            self.flush()
        } else {
            Ok(())
        }
    }

    pub fn flush(&mut self) -> std::io::Result<()> {
        if let Some(w) = self.writer.as_mut() {
            if !self.first {
                w.write(&self.cell)?;
                // clear cell byte
                self.cell[0] = 0;
            }
            self.first = true;
        }
        Ok(())
    }

    pub fn inner(mut self) -> std::io::Result<W> {
        self.flush()?;
        Ok(self.writer.take().unwrap())
    }

    pub fn cell_len_bits(&self) -> u32 {
        self.cell_len_bits
    }
}

impl<W: Write> Drop for CellWriter<W> {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

pub fn mem_address_proc_id_setup<W: Write>(
    cw: &mut CellWriter<W>,
    align_bits: u32,
    mem_address_end_pos: u64,
    proc_id_end_pos: u64,
) -> std::io::Result<()> {
    let cell_len_bits = cw.cell_len_bits();
    assert_ne!(mem_address_end_pos, 0);
    assert_ne!(proc_id_end_pos, 0);
    // cut cell_len_bits to max bits of these function's arguments (in this case 6 -> 64).
    let cell_len_bits_cut = std::cmp::min(6, cell_len_bits);
    let max_value = u64::try_from((1u128 << (1 << cell_len_bits_cut)) - 1).unwrap();
    let mut cell_count = 0u64;
    for value in [mem_address_end_pos, proc_id_end_pos] {
        let mut count = value;
        while count != 0 {
            let dec = std::cmp::min(max_value, count);
            // store
            cw.write_cell(dec)?;
            count -= dec;
            cell_count += 1;
        }
        cw.write_cell(0)?;
        cell_count += 1;
    }
    let align_mask = (1u64 << align_bits) - 1;
    if (cell_count & align_mask) != 0 {
        let extra_count = (1u64 << align_bits) - (cell_count & align_mask);
        for _ in 0..extra_count {
            cw.write_cell(0)?;
        }
    }
    Ok(())
}

// returns field_start and temp_buffer_step (length)
pub fn temp_buffer_first_field(
    data_part_len: u32,
    extra_end_pos_num: u32,
    field_num: u32,
) -> (u32, u32) {
    let field_start = ((2 + extra_end_pos_num) + data_part_len - 1) / data_part_len;
    (field_start, field_start + field_num)
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
    let (o, end, _, _) = par_process_infinite_data_stage(
        output_state,
        next_state,
        input,
        temp_buffer_step,
        &[(InfDataParam::ProcId, END_POS_PROC_ID)],
        &[(
            InfDataParam::TempBuffer(temp_buffer_step_pos),
            END_POS_PROC_ID,
        )],
        FuncNNAdapter1::from(Copy1Func::new()),
    );
    (o, end)
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
    let (o, end, _, _) = par_process_infinite_data_stage(
        output_state,
        next_state,
        input,
        temp_buffer_step,
        &[(InfDataParam::ProcId, END_POS_PROC_ID)],
        &[(InfDataParam::MemAddress, END_POS_MEM_ADDRESS)],
        FuncNNAdapter1::from(Copy1Func::new()),
    );
    (o, end)
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
    let (o, end, _, _) = par_process_infinite_data_stage(
        output_state,
        next_state,
        input,
        temp_buffer_step,
        &[(
            InfDataParam::TempBuffer(temp_buffer_step_pos),
            END_POS_MEM_ADDRESS,
        )],
        &[(InfDataParam::MemAddress, END_POS_MEM_ADDRESS)],
        FuncNNAdapter1::from(Copy1Func::new()),
    );
    (o, end)
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
    let (o, end, _, _) = par_process_infinite_data_stage(
        output_state,
        next_state,
        input,
        temp_buffer_step,
        &[(InfDataParam::MemAddress, END_POS_MEM_ADDRESS)],
        &[(
            InfDataParam::TempBuffer(temp_buffer_step_pos),
            END_POS_MEM_ADDRESS,
        )],
        FuncNNAdapter1::from(Copy1Func::new()),
    );
    (o, end)
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
    let end_pos = if proc_id_end_pos {
        END_POS_PROC_ID
    } else {
        END_POS_MEM_ADDRESS
    };
    let (o, end, _, _) = par_process_infinite_data_stage(
        output_state,
        next_state,
        input,
        temp_buffer_step,
        &[(InfDataParam::TempBuffer(tbs_src_pos), end_pos)],
        &[(InfDataParam::TempBuffer(tbs_dest_pos), end_pos)],
        FuncNNAdapter1::from(Copy1Func::new()),
    );
    (o, end)
}

// process routines

pub fn par_process_to_temp_buffer_stage<F: Function0>(
    output_state: UDynVarSys,
    next_state: UDynVarSys,
    input: &mut InfParInputSys,
    temp_buffer_step: u32,
    temp_buffer_step_pos: u32,
    proc_id_end_pos: bool,
    func: F,
) -> (InfParOutputSys, BoolVarSys, Vec<UDynVarSys>, BoolVarSys) {
    par_process_infinite_data_stage(
        output_state,
        next_state,
        input,
        temp_buffer_step,
        &[],
        &[(
            InfDataParam::TempBuffer(temp_buffer_step_pos),
            if proc_id_end_pos {
                END_POS_PROC_ID
            } else {
                END_POS_MEM_ADDRESS
            },
        )],
        FuncNNAdapter0::from(func),
    )
}

pub fn par_process_to_mem_address_stage<F: Function0>(
    output_state: UDynVarSys,
    next_state: UDynVarSys,
    input: &mut InfParInputSys,
    temp_buffer_step: u32,
    func: F,
) -> (InfParOutputSys, BoolVarSys, Vec<UDynVarSys>, BoolVarSys) {
    par_process_infinite_data_stage(
        output_state,
        next_state,
        input,
        temp_buffer_step,
        &[],
        &[(InfDataParam::MemAddress, END_POS_MEM_ADDRESS)],
        FuncNNAdapter0::from(func),
    )
}

// par_process_proc_id_to_temp_buffer_stage - process proc_id to temp buffer in specified pos.
// temp_buffer_step_pos - position chunk (specify data position).
pub fn par_process_proc_id_to_temp_buffer_stage<F: Function1>(
    output_state: UDynVarSys,
    next_state: UDynVarSys,
    input: &mut InfParInputSys,
    temp_buffer_step: u32,
    temp_buffer_step_pos: u32,
    func: F,
) -> (InfParOutputSys, BoolVarSys, Vec<UDynVarSys>, BoolVarSys) {
    par_process_infinite_data_stage(
        output_state,
        next_state,
        input,
        temp_buffer_step,
        &[(InfDataParam::ProcId, END_POS_PROC_ID)],
        &[(
            InfDataParam::TempBuffer(temp_buffer_step_pos),
            END_POS_PROC_ID,
        )],
        FuncNNAdapter1::from(func),
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
) -> (InfParOutputSys, BoolVarSys, Vec<UDynVarSys>, BoolVarSys) {
    par_process_infinite_data_stage(
        output_state,
        next_state,
        input,
        temp_buffer_step,
        &[(InfDataParam::ProcId, END_POS_PROC_ID)],
        &[(InfDataParam::MemAddress, END_POS_MEM_ADDRESS)],
        FuncNNAdapter1::from(func),
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
) -> (InfParOutputSys, BoolVarSys, Vec<UDynVarSys>, BoolVarSys) {
    par_process_infinite_data_stage(
        output_state,
        next_state,
        input,
        temp_buffer_step,
        &[(
            InfDataParam::TempBuffer(temp_buffer_step_pos),
            END_POS_MEM_ADDRESS,
        )],
        &[(InfDataParam::MemAddress, END_POS_MEM_ADDRESS)],
        FuncNNAdapter1::from(func),
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
) -> (InfParOutputSys, BoolVarSys, Vec<UDynVarSys>, BoolVarSys) {
    par_process_infinite_data_stage(
        output_state,
        next_state,
        input,
        temp_buffer_step,
        &[(InfDataParam::MemAddress, END_POS_MEM_ADDRESS)],
        &[(
            InfDataParam::TempBuffer(temp_buffer_step_pos),
            END_POS_MEM_ADDRESS,
        )],
        FuncNNAdapter1::from(func),
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
) -> (InfParOutputSys, BoolVarSys, Vec<UDynVarSys>, BoolVarSys) {
    par_process_infinite_data_stage(
        output_state,
        next_state,
        input,
        temp_buffer_step,
        &[(
            InfDataParam::TempBuffer(tbs_src_pos),
            if src_proc_id_end_pos {
                END_POS_PROC_ID
            } else {
                END_POS_MEM_ADDRESS
            },
        )],
        &[(
            InfDataParam::TempBuffer(tbs_dest_pos),
            if dest_proc_id_end_pos {
                END_POS_PROC_ID
            } else {
                END_POS_MEM_ADDRESS
            },
        )],
        FuncNNAdapter1::from(func),
    )
}

pub fn par_process_mem_address_stage<F: Function1>(
    output_state: UDynVarSys,
    next_state: UDynVarSys,
    input: &mut InfParInputSys,
    temp_buffer_step: u32,
    func: F,
) -> (InfParOutputSys, BoolVarSys, Vec<UDynVarSys>, BoolVarSys) {
    par_process_infinite_data_stage(
        output_state,
        next_state,
        input,
        temp_buffer_step,
        &[(InfDataParam::MemAddress, END_POS_MEM_ADDRESS)],
        &[(InfDataParam::MemAddress, END_POS_MEM_ADDRESS)],
        FuncNNAdapter1::from(func),
    )
}

// OP(proc_id, temp_buffer) -> temp_buffer
pub fn par_process_proc_id_temp_buffer_to_temp_buffer_stage<F: Function2>(
    output_state: UDynVarSys,
    next_state: UDynVarSys,
    input: &mut InfParInputSys,
    temp_buffer_step: u32,
    src_tbs_pos: u32,
    dest_tbs_pos: u32,
    src_proc_id_end_pos: bool,
    dest_proc_id_end_pos: bool,
    func: F,
) -> (InfParOutputSys, BoolVarSys, Vec<UDynVarSys>, BoolVarSys) {
    par_process_infinite_data_stage(
        output_state,
        next_state,
        input,
        temp_buffer_step,
        &[
            (InfDataParam::ProcId, END_POS_PROC_ID),
            (
                InfDataParam::TempBuffer(src_tbs_pos),
                if src_proc_id_end_pos {
                    END_POS_PROC_ID
                } else {
                    END_POS_MEM_ADDRESS
                },
            ),
        ],
        &[(
            InfDataParam::TempBuffer(dest_tbs_pos),
            if dest_proc_id_end_pos {
                END_POS_PROC_ID
            } else {
                END_POS_MEM_ADDRESS
            },
        )],
        FuncNNAdapter2::from(func),
    )
}

// OP(proc_id, mem_address) -> mem_address
pub fn par_process_proc_id_mem_address_to_mem_address_stage<F: Function2>(
    output_state: UDynVarSys,
    next_state: UDynVarSys,
    input: &mut InfParInputSys,
    temp_buffer_step: u32,
    func: F,
) -> (InfParOutputSys, BoolVarSys, Vec<UDynVarSys>, BoolVarSys) {
    par_process_infinite_data_stage(
        output_state,
        next_state,
        input,
        temp_buffer_step,
        &[
            (InfDataParam::ProcId, END_POS_PROC_ID),
            (InfDataParam::MemAddress, END_POS_MEM_ADDRESS),
        ],
        &[(InfDataParam::MemAddress, END_POS_MEM_ADDRESS)],
        FuncNNAdapter2::from(func),
    )
}

// OP(proc_id, mem_address) -> temp_buffer
pub fn par_process_proc_id_mem_address_to_temp_buffer_stage<F: Function2>(
    output_state: UDynVarSys,
    next_state: UDynVarSys,
    input: &mut InfParInputSys,
    temp_buffer_step: u32,
    dest_tbs_pos: u32,
    dest_proc_id_end_pos: bool,
    func: F,
) -> (InfParOutputSys, BoolVarSys, Vec<UDynVarSys>, BoolVarSys) {
    par_process_infinite_data_stage(
        output_state,
        next_state,
        input,
        temp_buffer_step,
        &[
            (InfDataParam::ProcId, END_POS_PROC_ID),
            (InfDataParam::MemAddress, END_POS_MEM_ADDRESS),
        ],
        &[(
            InfDataParam::TempBuffer(dest_tbs_pos),
            if dest_proc_id_end_pos {
                END_POS_PROC_ID
            } else {
                END_POS_MEM_ADDRESS
            },
        )],
        FuncNNAdapter2::from(func),
    )
}

// OP(proc_id, temp_buffer) -> mem_address
pub fn par_process_proc_id_temp_buffer_to_mem_address_stage<F: Function2>(
    output_state: UDynVarSys,
    next_state: UDynVarSys,
    input: &mut InfParInputSys,
    temp_buffer_step: u32,
    src_tbs_pos: u32,
    src_proc_id_end_pos: bool,
    func: F,
) -> (InfParOutputSys, BoolVarSys, Vec<UDynVarSys>, BoolVarSys) {
    par_process_infinite_data_stage(
        output_state,
        next_state,
        input,
        temp_buffer_step,
        &[
            (InfDataParam::ProcId, END_POS_PROC_ID),
            (
                InfDataParam::TempBuffer(src_tbs_pos),
                if src_proc_id_end_pos {
                    END_POS_PROC_ID
                } else {
                    END_POS_MEM_ADDRESS
                },
            ),
        ],
        &[(InfDataParam::MemAddress, END_POS_MEM_ADDRESS)],
        FuncNNAdapter2::from(func),
    )
}

// OP(mem_address, temp_buffer) -> temp_buffer
pub fn par_process_mem_address_temp_buffer_to_temp_buffer_stage<F: Function2>(
    output_state: UDynVarSys,
    next_state: UDynVarSys,
    input: &mut InfParInputSys,
    temp_buffer_step: u32,
    src_tbs_pos: u32,
    dest_tbs_pos: u32,
    src_proc_id_end_pos: bool,
    dest_proc_id_end_pos: bool,
    func: F,
) -> (InfParOutputSys, BoolVarSys, Vec<UDynVarSys>, BoolVarSys) {
    par_process_infinite_data_stage(
        output_state,
        next_state,
        input,
        temp_buffer_step,
        &[
            (InfDataParam::MemAddress, END_POS_MEM_ADDRESS),
            (
                InfDataParam::TempBuffer(src_tbs_pos),
                if src_proc_id_end_pos {
                    END_POS_PROC_ID
                } else {
                    END_POS_MEM_ADDRESS
                },
            ),
        ],
        &[(
            InfDataParam::TempBuffer(dest_tbs_pos),
            if dest_proc_id_end_pos {
                END_POS_PROC_ID
            } else {
                END_POS_MEM_ADDRESS
            },
        )],
        FuncNNAdapter2::from(func),
    )
}

// OP(mem_address, temp_buffer) -> mem_address
pub fn par_process_mem_address_temp_buffer_to_mem_address_stage<F: Function2>(
    output_state: UDynVarSys,
    next_state: UDynVarSys,
    input: &mut InfParInputSys,
    temp_buffer_step: u32,
    src_tbs_pos: u32,
    src_proc_id_end_pos: bool,
    func: F,
) -> (InfParOutputSys, BoolVarSys, Vec<UDynVarSys>, BoolVarSys) {
    par_process_infinite_data_stage(
        output_state,
        next_state,
        input,
        temp_buffer_step,
        &[
            (InfDataParam::MemAddress, END_POS_MEM_ADDRESS),
            (
                InfDataParam::TempBuffer(src_tbs_pos),
                if src_proc_id_end_pos {
                    END_POS_PROC_ID
                } else {
                    END_POS_MEM_ADDRESS
                },
            ),
        ],
        &[(InfDataParam::MemAddress, END_POS_MEM_ADDRESS)],
        FuncNNAdapter2::from(func),
    )
}

// OP(temp_buffer, temp_buffer) -> temp_buffer
pub fn par_process_temp_buffer_temp_buffer_to_temp_buffer_stage<F: Function2>(
    output_state: UDynVarSys,
    next_state: UDynVarSys,
    input: &mut InfParInputSys,
    temp_buffer_step: u32,
    src_tbs_pos: u32,
    src2_tbs_pos: u32,
    dest_tbs_pos: u32,
    src_proc_id_end_pos: bool,
    src2_proc_id_end_pos: bool,
    dest_proc_id_end_pos: bool,
    func: F,
) -> (InfParOutputSys, BoolVarSys, Vec<UDynVarSys>, BoolVarSys) {
    par_process_infinite_data_stage(
        output_state,
        next_state,
        input,
        temp_buffer_step,
        &[
            (
                InfDataParam::TempBuffer(src_tbs_pos),
                if src_proc_id_end_pos {
                    END_POS_PROC_ID
                } else {
                    END_POS_MEM_ADDRESS
                },
            ),
            (
                InfDataParam::TempBuffer(src2_tbs_pos),
                if src2_proc_id_end_pos {
                    END_POS_PROC_ID
                } else {
                    END_POS_MEM_ADDRESS
                },
            ),
        ],
        &[(
            InfDataParam::TempBuffer(dest_tbs_pos),
            if dest_proc_id_end_pos {
                END_POS_PROC_ID
            } else {
                END_POS_MEM_ADDRESS
            },
        )],
        FuncNNAdapter2::from(func),
    )
}

// OP(temp_buffer, temp_buffer) -> mem_address
pub fn par_process_temp_buffer_2_to_mem_address_stage<F: Function2>(
    output_state: UDynVarSys,
    next_state: UDynVarSys,
    input: &mut InfParInputSys,
    temp_buffer_step: u32,
    src_tbs_pos: u32,
    src2_tbs_pos: u32,
    src_proc_id_end_pos: bool,
    src2_proc_id_end_pos: bool,
    func: F,
) -> (InfParOutputSys, BoolVarSys, Vec<UDynVarSys>, BoolVarSys) {
    par_process_infinite_data_stage(
        output_state,
        next_state,
        input,
        temp_buffer_step,
        &[
            (
                InfDataParam::TempBuffer(src_tbs_pos),
                if src_proc_id_end_pos {
                    END_POS_PROC_ID
                } else {
                    END_POS_MEM_ADDRESS
                },
            ),
            (
                InfDataParam::TempBuffer(src2_tbs_pos),
                if src2_proc_id_end_pos {
                    END_POS_PROC_ID
                } else {
                    END_POS_MEM_ADDRESS
                },
            ),
        ],
        &[(InfDataParam::MemAddress, END_POS_MEM_ADDRESS)],
        FuncNNAdapter2::from(func),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mem_address_proc_id_setup_helper(
        cell_len_bits: u32,
        align_bits: u32,
        mem_address_end_pos: u64,
        proc_id_end_pos: u64,
    ) -> Vec<u8> {
        let out = vec![];
        let mut cw = CellWriter::new(cell_len_bits, out);
        mem_address_proc_id_setup(&mut cw, align_bits, mem_address_end_pos, proc_id_end_pos)
            .unwrap();
        cw.inner().unwrap()
    }

    #[test]
    fn test_mem_address_proc_id_setup() {
        // 1-bit cell
        assert_eq!(
            vec![0b11111111, 0b11111111, 0b11111011, 0b11111111, 0b11111111, 0b00111111],
            mem_address_proc_id_setup_helper(0, 3, 18, 27)
        );
        assert_eq!(
            vec![0b11111111, 0b11111111, 0b11111011, 0b11111111, 0b11111111, 0b01111111],
            mem_address_proc_id_setup_helper(0, 3, 18, 28)
        );
        assert_eq!(
            vec![0b11111111, 0b11111111, 0b11111011, 0b11111111, 0b11111111, 0b01111111, 0, 0],
            mem_address_proc_id_setup_helper(0, 5, 18, 28)
        );
        assert_eq!(
            vec![0b11111111, 0b11111111, 0b11111011, 0b11111111, 0b11111111, 0b11111111, 0],
            mem_address_proc_id_setup_helper(0, 3, 18, 29)
        );
        // 2-bit cell
        assert_eq!(
            vec![0b11111111, 0b11001011, 0b11111111, 0b11111111, 0b0001],
            mem_address_proc_id_setup_helper(1, 2, 17, 28)
        );
        assert_eq!(
            vec![0b11111111, 0b11001011, 0b11111111, 0b11111111, 0b0001, 0],
            mem_address_proc_id_setup_helper(1, 3, 17, 28)
        );
        assert_eq!(
            vec![0b11111111, 0b11001011, 0b11111111, 0b11111111, 0b0001, 0, 0, 0],
            mem_address_proc_id_setup_helper(1, 4, 17, 28)
        );
        // 4-bit cell
        assert_eq!(
            vec![0x2f, 0xf0, 0xd],
            mem_address_proc_id_setup_helper(2, 1, 17, 28)
        );
        assert_eq!(
            vec![0x2f, 0xf0, 0xd, 0],
            mem_address_proc_id_setup_helper(2, 2, 17, 28)
        );
        assert_eq!(
            vec![0xff, 0x4f, 0xf0, 0xff, 0xff, 0x5],
            mem_address_proc_id_setup_helper(2, 1, 49, 80)
        );
        assert_eq!(
            vec![0xff, 0x4f, 0xf0, 0xff, 0xff, 0x5, 0, 0],
            mem_address_proc_id_setup_helper(2, 3, 49, 80)
        );
        // 4-bit cell
        assert_eq!(
            vec![17, 0, 28, 0],
            mem_address_proc_id_setup_helper(3, 0, 17, 28)
        );
        assert_eq!(
            vec![17, 0, 28, 0, 0, 0, 0, 0],
            mem_address_proc_id_setup_helper(3, 3, 17, 28)
        );
        // 8-bit cell
        assert_eq!(
            vec![255, 255, 241, 0, 255, 255, 255, 255, 141, 0],
            mem_address_proc_id_setup_helper(3, 0, 751, 1161)
        );
        assert_eq!(
            vec![255, 255, 241, 0, 255, 255, 255, 255, 141, 0, 0, 0],
            mem_address_proc_id_setup_helper(3, 2, 751, 1161)
        );
        // 16-bit cell
        assert_eq!(
            vec![0xef, 2, 0, 0, 0x89, 4, 0, 0],
            mem_address_proc_id_setup_helper(4, 0, 751, 1161)
        );
        assert_eq!(
            vec![0xef, 2, 0, 0, 0x89, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            mem_address_proc_id_setup_helper(4, 3, 751, 1161)
        );
        assert_eq!(
            vec![
                255, 255, 255, 255, 255, 255, 255, 255, 0xaf, 0x32, 0, 0, 255, 255, 255, 255, 255,
                255, 0xa2, 0x3a, 0, 0
            ],
            mem_address_proc_id_setup_helper(4, 0, 275115, 211615)
        );
        assert_eq!(
            vec![
                255, 255, 255, 255, 255, 255, 255, 255, 0xaf, 0x32, 0, 0, 255, 255, 255, 255, 255,
                255, 0xa2, 0x3a, 0, 0, 0, 0
            ],
            mem_address_proc_id_setup_helper(4, 2, 275115, 211615)
        );
        // 32-bit cell
        assert_eq!(
            vec![0xab, 0x32, 0x4, 0, 0, 0, 0, 0, 0x9f, 0x3a, 0x3, 0, 0, 0, 0, 0],
            mem_address_proc_id_setup_helper(5, 0, 275115, 211615)
        );
        // 64-bit cell
        assert_eq!(
            vec![
                0xab, 0x32, 0x4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x9f, 0x3a, 0x3, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ],
            mem_address_proc_id_setup_helper(6, 0, 275115, 211615)
        );
        // 128-bit cell
        assert_eq!(
            vec![
                0xab, 0x32, 0x4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0x9f, 0x3a, 0x3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
            mem_address_proc_id_setup_helper(7, 0, 275115, 211615)
        );
    }

    fn cell_writer_helper(cell_len_bits: u32, iter: impl IntoIterator<Item = u64>) -> Vec<u8> {
        let out = vec![];
        let mut cw = CellWriter::new(cell_len_bits, out);
        for v in iter.into_iter() {
            cw.write_cell(v).unwrap();
        }
        cw.inner().unwrap()
    }

    #[test]
    fn test_cell_writer() {
        assert_eq!(vec![0b1101], cell_writer_helper(0, [1, 0, 1, 1]));
        assert_eq!(
            vec![0b10001101],
            cell_writer_helper(0, [1, 0, 1, 1, 0, 0, 0, 1])
        );
        assert_eq!(
            vec![0b10001101, 0b1011],
            cell_writer_helper(0, [1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1])
        );
        assert_eq!(
            vec![0b00111001, 0b011011],
            cell_writer_helper(1, [1, 2, 3, 0, 3, 2, 1])
        );
        assert_eq!(
            vec![0x37, 0xb8, 0xed],
            cell_writer_helper(2, [7, 3, 8, 11, 13, 14])
        );
        assert_eq!(
            vec![0x37, 0xb8, 0xad, 0x02],
            cell_writer_helper(2, [7, 3, 8, 11, 13, 10, 2])
        );
        assert_eq!(
            vec![55, 67, 86, 11],
            cell_writer_helper(3, [55, 67, 86, 11])
        );
        assert_eq!(
            vec![55, 67, 86, 11, 119],
            cell_writer_helper(3, [55, 67, 86, 11, 119])
        );
        assert_eq!(
            vec![0xcd, 0xab, 0xca, 0x04, 0x91, 0x9],
            cell_writer_helper(4, [0xabcd, 0x4ca, 0x991])
        );
        assert_eq!(
            vec![0xcd, 0xab, 0x11, 0, 0xca, 0x04, 0x77, 0, 0x91, 0x19, 0xd1, 0x12],
            cell_writer_helper(5, [0x11abcd, 0x7704ca, 0x12d11991])
        );
        assert_eq!(
            vec![
                0xcd, 0xab, 0x11, 0, 0x6, 0, 0, 0, 0xca, 0x04, 0x77, 0x10, 0x22, 0, 0, 0, 0x91,
                0x19, 0xd1, 0x12, 0, 0x66, 0, 0
            ],
            cell_writer_helper(6, [0x60011abcd, 0x22107704ca, 0x660012d11991])
        );
        assert_eq!(
            vec![
                0xcd, 0xab, 0x11, 0, 0x6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xca, 0x04, 0x77, 0x10,
                0x22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x91, 0x19, 0xd1, 0x12, 0, 0x66, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
            ],
            cell_writer_helper(7, [0x60011abcd, 0x22107704ca, 0x660012d11991])
        );
    }
}
