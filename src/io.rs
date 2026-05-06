use csv::Writer;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// Extracts filename without extension.
pub fn filename_only(path: &Path) -> String {
    path.file_stem().unwrap().to_string_lossy().to_string()
}

/// Writes persistence images to CSV.
///
/// Format:
/// [filename, pixel_1, pixel_2, ...]
pub fn save_images_to_csv(
    filenames: &[String],
    images: &[Vec<f64>],
    output_path: &Path,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let file = File::create(output_path)?;
    let mut wtr = Writer::from_writer(file);

    for (name, image) in filenames.iter().zip(images.iter()) {
        let mut row = Vec::with_capacity(image.len() + 1);

        row.push(name.clone());

        for &val in image {
            row.push(val.to_string());
        }

        wtr.write_record(&row)?;
    }

    wtr.flush()?;
    Ok(())
}

/// Recursively collects all CSV files in a directory.
pub fn collect_csv_files<P: AsRef<Path>>(root: P) -> Vec<PathBuf> {
    WalkDir::new(root)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| {
            e.file_type().is_file() && e.path().extension().map_or(false, |ext| ext == "csv")
        })
        .map(|e| e.into_path())
        .collect()
}

/// Loads voxel spacing map from CSV.
///
/// Expected format:
/// filename, ..., voxel_spacing (column 2)
pub fn load_spacing<P: AsRef<Path>>(
    path: P,
) -> Result<HashMap<String, f64>, Box<dyn Error + Send + Sync>> {
    let file = File::open(path)?;
    let mut rdr = csv::Reader::from_reader(file);

    let mut map = HashMap::new();

    for result in rdr.records() {
        let record = result?;

        let filename = record.get(0).ok_or("Missing filename column")?.to_string();

        let voxel_spacing: f64 = record.get(2).ok_or("Missing spacing column")?.parse()?;

        map.insert(filename, voxel_spacing);
    }

    Ok(map)
}

/// Extracts base filename (without extension).
pub fn base_name(path: &Path) -> Option<String> {
    path.file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.to_string())
}
