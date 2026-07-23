//! PyO3 binding methods split by domain.

use super::*;

#[pymethods]
impl ApexStorageImpl {
    fn read_blob(&self, py: Python<'_>, column: String, id: i64) -> PyResult<Option<PyObject>> {
        use pyo3::types::PyBytes;

        if id < 0 || column.is_empty() {
            return Ok(None);
        }
        let backend = self.get_read_backend_cached(py)?;
        let bytes = py
            .allow_threads(|| backend.storage.read_blob_by_id(&column, id as u64))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(bytes.map(|b| PyBytes::new_bound(py, &b).into()))
    }

    fn read_blobs(&self, py: Python<'_>, column: String, ids: Vec<i64>) -> PyResult<PyObject> {
        use pyo3::types::{PyBytes, PyList};

        let out = PyList::empty_bound(py);
        if column.is_empty() || ids.is_empty() {
            return Ok(out.into());
        }

        let mut positions = Vec::new();
        let mut valid_ids = Vec::new();
        for (pos, id) in ids.iter().enumerate() {
            if *id >= 0 {
                positions.push(pos);
                valid_ids.push(*id as u64);
            }
        }

        let mut values: Vec<Option<Vec<u8>>> = vec![None; ids.len()];
        if !valid_ids.is_empty() {
            let backend = self.get_read_backend_cached(py)?;
            let blobs = py
                .allow_threads(|| backend.storage.read_blobs_by_ids(&column, &valid_ids))
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            for (pos, blob) in positions.into_iter().zip(blobs.into_iter()) {
                values[pos] = blob;
            }
        }

        for value in values {
            if let Some(bytes) = value {
                out.append(PyBytes::new_bound(py, &bytes))?;
            } else {
                out.append(py.None())?;
            }
        }
        Ok(out.into())
    }

    #[pyo3(signature = (column, id, offset, length=None))]
    fn read_blob_range(
        &self,
        py: Python<'_>,
        column: String,
        id: i64,
        offset: u64,
        length: Option<usize>,
    ) -> PyResult<Option<PyObject>> {
        use pyo3::types::PyBytes;

        if id < 0 || column.is_empty() {
            return Ok(None);
        }
        let backend = self.get_read_backend_cached(py)?;
        let bytes = py
            .allow_threads(|| {
                backend
                    .storage
                    .read_blob_range_by_id(&column, id as u64, offset, length)
            })
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(bytes.map(|b| PyBytes::new_bound(py, &b).into()))
    }

    #[pyo3(signature = (column, ids, offsets, length=None))]
    fn read_blob_ranges(
        &self,
        py: Python<'_>,
        column: String,
        ids: Vec<i64>,
        offsets: Vec<u64>,
        length: Option<usize>,
    ) -> PyResult<PyObject> {
        use pyo3::types::{PyBytes, PyList};

        if ids.len() != offsets.len() {
            return Err(PyValueError::new_err(
                "ids and offsets must have the same length",
            ));
        }

        let out = PyList::empty_bound(py);
        if column.is_empty() || ids.is_empty() {
            return Ok(out.into());
        }

        let mut positions = Vec::new();
        let mut valid_ids = Vec::new();
        let mut valid_offsets = Vec::new();
        for (pos, id) in ids.iter().enumerate() {
            if *id >= 0 {
                positions.push(pos);
                valid_ids.push(*id as u64);
                valid_offsets.push(offsets[pos]);
            }
        }

        let mut values: Vec<Option<Vec<u8>>> = vec![None; ids.len()];
        if !valid_ids.is_empty() {
            let backend = self.get_read_backend_cached(py)?;
            let blobs = py
                .allow_threads(|| {
                    backend.storage.read_blob_ranges_by_ids(
                        &column,
                        &valid_ids,
                        &valid_offsets,
                        length,
                    )
                })
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            for (pos, blob) in positions.into_iter().zip(blobs.into_iter()) {
                values[pos] = blob;
            }
        }

        for value in values {
            if let Some(bytes) = value {
                out.append(PyBytes::new_bound(py, &bytes))?;
            } else {
                out.append(py.None())?;
            }
        }
        Ok(out.into())
    }

    fn read_blob_descriptor(
        &self,
        py: Python<'_>,
        column: String,
        id: i64,
    ) -> PyResult<Option<PyObject>> {
        use pyo3::types::PyBytes;

        if id < 0 || column.is_empty() {
            return Ok(None);
        }
        let backend = self.get_read_backend_cached(py)?;
        let descriptor = py
            .allow_threads(|| {
                backend
                    .storage
                    .read_blob_descriptor_by_id(&column, id as u64)
            })
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(descriptor.map(|b| PyBytes::new_bound(py, &b).into()))
    }

    fn read_blob_info(
        &self,
        py: Python<'_>,
        column: String,
        id: i64,
    ) -> PyResult<Option<PyObject>> {
        if id < 0 || column.is_empty() {
            return Ok(None);
        }
        let backend = self.get_read_backend_cached(py)?;
        let info = py
            .allow_threads(|| {
                backend
                    .storage
                    .read_blob_descriptor_info_by_id(&column, id as u64)
            })
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        let Some(info) = info else {
            return Ok(None);
        };

        let dict = PyDict::new_bound(py);
        dict.set_item("mode", Self::blob_mode_to_str(info.mode))?;
        dict.set_item("length", info.len)?;
        dict.set_item("checksum", info.checksum)?;
        dict.set_item("locator_length", info.locator_len)?;
        Ok(Some(dict.into()))
    }

    fn read_blob_infos(&self, py: Python<'_>, column: String, ids: Vec<i64>) -> PyResult<PyObject> {
        use pyo3::types::PyList;

        let out = PyList::empty_bound(py);
        if column.is_empty() || ids.is_empty() {
            return Ok(out.into());
        }

        let mut positions = Vec::new();
        let mut valid_ids = Vec::new();
        for (pos, id) in ids.iter().enumerate() {
            if *id >= 0 {
                positions.push(pos);
                valid_ids.push(*id as u64);
            }
        }

        let mut infos: Vec<Option<crate::storage::on_demand::BlobDescriptorInfo>> =
            vec![None; ids.len()];
        if !valid_ids.is_empty() {
            let backend = self.get_read_backend_cached(py)?;
            let read_infos = py
                .allow_threads(|| {
                    backend
                        .storage
                        .read_blob_descriptor_infos_by_ids(&column, &valid_ids)
                })
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
            for (pos, info) in positions.into_iter().zip(read_infos.into_iter()) {
                infos[pos] = info;
            }
        }

        for info in infos {
            if let Some(info) = info {
                let dict = PyDict::new_bound(py);
                dict.set_item("mode", Self::blob_mode_to_str(info.mode))?;
                dict.set_item("length", info.len)?;
                dict.set_item("checksum", info.checksum)?;
                dict.set_item("locator_length", info.locator_len)?;
                out.append(dict)?;
            } else {
                out.append(py.None())?;
            }
        }
        Ok(out.into())
    }
}
