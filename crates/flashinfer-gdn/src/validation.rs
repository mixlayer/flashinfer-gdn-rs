use crate::{CudaTensor, DType, Error, Result};

pub(crate) fn bounds_overlap(left: (usize, usize), right: (usize, usize)) -> bool {
    left.0 < right.1 && right.0 < left.1
}

pub(crate) fn check_rank(tensor: &CudaTensor, name: &'static str, rank: usize) -> Result<()> {
    if tensor.shape().len() != rank {
        return Err(Error::tensor(
            name,
            format!("expected rank {rank}, found shape {:?}", tensor.shape()),
        ));
    }
    Ok(())
}

pub(crate) fn expect(
    tensor: &CudaTensor,
    name: &'static str,
    dtype: DType,
    shape: &[i64],
    device_id: i32,
) -> Result<()> {
    if tensor.dtype() != dtype {
        return Err(Error::tensor(
            name,
            format!("expected dtype {dtype:?}, found {:?}", tensor.dtype()),
        ));
    }
    if tensor.shape() != shape {
        return Err(Error::tensor(
            name,
            format!("expected shape {shape:?}, found {:?}", tensor.shape()),
        ));
    }
    if tensor.device_id() != device_id {
        return Err(Error::tensor(
            name,
            format!(
                "expected CUDA device {device_id}, found {}",
                tensor.device_id()
            ),
        ));
    }
    if tensor.byte_offset() != 0 {
        return Err(Error::tensor(
            name,
            format!(
                "generated GDN entrypoints require byte_offset=0, found {}",
                tensor.byte_offset()
            ),
        ));
    }
    if tensor.strides().last() != Some(&1) {
        return Err(Error::tensor(
            name,
            format!(
                "innermost dimension must be contiguous, found strides {:?}",
                tensor.strides()
            ),
        ));
    }
    Ok(())
}
