use crate::diagram::{Diagram, read_diagram};
use crate::io::{base_name, collect_csv_files, load_spacing};
use rayon::prelude::*;
use std::error::Error;
use std::path::{Path, PathBuf};

/// A persistence diagram together with its source path
/// and voxel spacing metadata.
#[derive(Clone)]
pub struct SpacingDiagram {
    pub path: PathBuf,
    pub diagram: Diagram,
    pub voxel_spacing: f64,
}

/// Loads the dataset:
/// - scans directory for CSV diagrams
/// - loads spacing metadata
/// - attaches spacing to each diagram
pub fn load_dataset<P: AsRef<Path>>(
    root: P,
    spacing_csv: P,
    remove_h0: bool,
) -> Result<Vec<SpacingDiagram>, Box<dyn Error + Send + Sync>> {
    let spacing_map = load_spacing(spacing_csv)?;

    let mut dataset = Vec::new();

    for path in collect_csv_files(root) {
        let diagram = read_diagram(&path, remove_h0)?;

        let key = base_name(&path).ok_or("Invalid filename")?;

        let voxel_spacing = spacing_map
            .get(&format!("{key}.tif"))
            .copied()
            .unwrap_or(1.0);

        dataset.push(SpacingDiagram {
            path: path.to_path_buf(),
            diagram,
            voxel_spacing,
        });
    }

    Ok(dataset)
}

/// Applies voxel spacing to a single diagram.
/// Returns a *pure* Diagram (spacing absorbed).
pub fn apply_spacing(sample: &SpacingDiagram) -> Diagram {
    sample
        .diagram
        .iter()
        .map(|&(b, p)| (b * sample.voxel_spacing, p * sample.voxel_spacing))
        .collect()
}

/// Applies spacing to the whole dataset in parallel.
pub fn to_scaled_diagrams(dataset: &[SpacingDiagram]) -> Vec<Diagram> {
    dataset.par_iter().map(apply_spacing).collect()
}
