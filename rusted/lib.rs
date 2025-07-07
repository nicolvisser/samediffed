use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use rayon::prelude::*;

pub fn edit_distance(a: &[u32], b: &[u32]) -> usize {
    let len_a = a.len();
    let len_b = b.len();
    if len_a < len_b {
        return edit_distance(b, a);
    }
    // handle special case of 0 length
    if len_a == 0 {
        return len_b;
    } else if len_b == 0 {
        return len_a;
    }

    let len_b = len_b + 1;

    let mut pre;
    let mut tmp;
    let mut cur = vec![0; len_b];

    // initialize sequence b
    for i in 1..len_b {
        cur[i] = i;
    }

    // calculate edit distance
    for (i, &ca) in a.iter().enumerate() {
        // get first column for this row
        pre = cur[0];
        cur[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            tmp = cur[j + 1];
            cur[j + 1] = std::cmp::min(
                // deletion
                tmp + 1,
                std::cmp::min(
                    // insertion
                    cur[j] + 1,
                    // match or substitution
                    pre + if ca == cb { 0 } else { 1 },
                ),
            );
            pre = tmp;
        }
    }
    cur[len_b - 1]
}

/// Python wrapper for edit_distance function
#[pyfunction]
fn py_edit_distance(a: Vec<u32>, b: Vec<u32>) -> PyResult<usize> {
    Ok(edit_distance(&a, &b))
}

/// Function to calculate normalized edit distance (edit distance / max length)
pub fn normalized_edit_distance(a: &[u32], b: &[u32]) -> f64 {
    let edit_dist = edit_distance(a, b);
    let max_len = std::cmp::max(a.len(), b.len());
    if max_len == 0 {
        0.0
    } else {
        edit_dist as f64 / max_len as f64
    }
}

/// Python wrapper for normalized_edit_distance function
#[pyfunction]
fn py_normalized_edit_distance(a: Vec<u32>, b: Vec<u32>) -> PyResult<f64> {
    Ok(normalized_edit_distance(&a, &b))
}

/// Python module definition
#[pymodule]
fn _rusted(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_edit_distance, m)?)?;
    m.add_function(wrap_pyfunction!(edit_distance_polars, m)?)?;
    m.add_function(wrap_pyfunction!(py_normalized_edit_distance, m)?)?;
    m.add_function(wrap_pyfunction!(normalized_edit_distance_polars, m)?)?;
    Ok(())
}

/// Function to calculate edit distance for Polars DataFrame columns
#[pyfunction]
fn edit_distance_polars(seq1: Vec<Vec<u32>>, seq2: Vec<Vec<u32>>) -> PyResult<Vec<usize>> {
    if seq1.len() != seq2.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Sequences must have the same length"
        ));
    }
    // Parallelize the computation
    let results: Vec<usize> = seq1.par_iter().zip(seq2.par_iter())
        .map(|(a, b)| edit_distance(a, b))
        .collect();
    Ok(results)
}

/// Function to calculate normalized edit distance for Polars DataFrame columns
#[pyfunction]
fn normalized_edit_distance_polars(seq1: Vec<Vec<u32>>, seq2: Vec<Vec<u32>>) -> PyResult<Vec<f64>> {
    if seq1.len() != seq2.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Sequences must have the same length"
        ));
    }
    // Parallelize the computation
    let results: Vec<f64> = seq1.par_iter().zip(seq2.par_iter())
        .map(|(a, b)| normalized_edit_distance(a, b))
        .collect();
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edit_distance() {
        assert_eq!(edit_distance(&[1, 2, 3], &[1, 2, 3]), 0);
        assert_eq!(edit_distance(&[1, 2, 3], &[1, 2, 4]), 1);
        assert_eq!(edit_distance(&[1, 2, 3], &[1, 2]), 1);
        assert_eq!(edit_distance(&[1, 2], &[1, 2, 3]), 1);
        
        assert_eq!(edit_distance(&[], &[1, 2, 3]), 3);
        assert_eq!(edit_distance(&[1, 2, 3], &[]), 3);
        assert_eq!(edit_distance(&[], &[]), 0);
        
        assert_eq!(edit_distance(&[1, 2, 3, 4, 5], &[1, 2, 4, 5]), 1);
        assert_eq!(edit_distance(&[1, 2, 3, 4], &[1, 3, 4, 5]), 2);
    }
}
