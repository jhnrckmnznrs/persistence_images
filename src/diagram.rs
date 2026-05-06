use std::error::Error;
use std::fs::File;
use std::path::Path;

/// A persistence diagram represented as (birth, persistence) pairs.
pub type Diagram = Vec<(f64, f64)>;

/// Reads a persistence diagram from a CSV file.
///
/// The input CSV is expected to have at least two columns:
/// - column 0: birth
/// - column 1: death
///
/// # Arguments
/// - `path`: Path to the CSV file
/// - `remove_h0`: Whether to remove the dominant H0 class (longest-lived component)
///
/// # Returns
/// A vector of (birth, persistence) pairs.
///
/// # Errors
/// Returns an error if:
/// - the file cannot be opened
/// - parsing fails
pub fn read_diagram(path: &Path, remove_h0: bool) -> Result<Diagram, Box<dyn Error + Send + Sync>> {
    let file = File::open(path)?;
    let mut rdr = csv::Reader::from_reader(file);

    let mut data = Vec::new();

    for (i, result) in rdr.records().enumerate() {
        let record = result?;

        let birth: f64 = record
            .get(0)
            .ok_or_else(|| format!("Missing birth value at row {}", i))?
            .parse()?;

        let death: f64 = record
            .get(1)
            .ok_or_else(|| format!("Missing death value at row {}", i))?
            .parse()?;

        data.push((birth, death));
    }

    let data = if remove_h0 {
        remove_infinite_class(&data)
    } else {
        data
    };

    // Convert (birth, death) → (birth, persistence)
    let diagram = data.into_iter().map(|(b, d)| (b, d - b)).collect();

    Ok(diagram)
}

/// Removes the dominant H0 class from a persistence diagram.
///
/// This typically corresponds to the connected component with:
/// - smallest birth
/// - largest death (in case of ties)
///
/// # Arguments
/// - `diagram`: Slice of (birth, death) pairs
///
/// # Returns
/// A new diagram with the selected point removed.
///
/// # Notes
/// This is commonly used to discard the "infinite" connected component.
pub fn remove_infinite_class(diagram: &[(f64, f64)]) -> Vec<(f64, f64)> {
    if diagram.is_empty() {
        return Vec::new();
    }

    let mut best_idx = 0;
    let mut best_birth = diagram[0].0;
    let mut best_death = diagram[0].1;

    for (i, &(birth, death)) in diagram.iter().enumerate() {
        if birth < best_birth || (birth == best_birth && death > best_death) {
            best_birth = birth;
            best_death = death;
            best_idx = i;
        }
    }

    diagram
        .iter()
        .enumerate()
        .filter_map(|(i, &point)| if i == best_idx { None } else { Some(point) })
        .collect()
}
