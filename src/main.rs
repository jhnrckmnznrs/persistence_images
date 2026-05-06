use clap::Parser;
use std::error::Error;
use std::path::PathBuf;

use persistence_images::{
    dataset::{SpacingDiagram, load_dataset, to_scaled_diagrams},
    image::{compute_bounds, persistence_image},
    io,
};

use rayon::prelude::*;

/// Persistence Image Generator from Persistence Diagrams
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Input directory containing persistence diagram CSV files
    #[arg(short, long)]
    input: PathBuf,

    /// CSV file containing voxel spacing information
    #[arg(short, long)]
    spacing: PathBuf,

    /// Output CSV file path
    #[arg(short, long)]
    output: PathBuf,

    /// Grid resolution in x direction
    #[arg(long, default_value_t = 10)]
    nx: usize,

    /// Grid resolution in y direction
    #[arg(long, default_value_t = 8)]
    ny: usize,

    /// Global sigma x
    #[arg(long, default_value_t = 0.86)]
    sigma_x: f64,

    /// Global sigma y
    #[arg(long, default_value_t = 0.63984)]
    sigma_y: f64,

    /// Remove H0 component
    #[arg(long, default_value_t = true)]
    remove_h0: bool,
}

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let args = Args::parse();

    // -----------------------------
    // 1. Load dataset
    // -----------------------------
    let dataset: Vec<SpacingDiagram> = load_dataset(&args.input, &args.spacing, args.remove_h0)?;

    if dataset.is_empty() {
        println!("No data found.");
        return Ok(());
    }

    // -----------------------------
    // 2. Apply spacing
    // -----------------------------
    let diagrams = to_scaled_diagrams(&dataset);

    // -----------------------------
    // 3. Compute bounds
    // -----------------------------
    let bounds = compute_bounds(&diagrams).expect("Failed to compute bounds (empty dataset)");

    println!("Bounds: {:?}", bounds);

    // -----------------------------
    // 4. Compute persistence images
    // -----------------------------
    let images: Vec<Vec<f64>> = dataset
        .par_iter()
        .zip(diagrams.par_iter())
        .map(|(sample, diagram)| {
            let sigma_x_local = args.sigma_x * sample.voxel_spacing;
            let sigma_y_local = args.sigma_y * sample.voxel_spacing;

            persistence_image(
                diagram,
                bounds,
                args.nx,
                args.ny,
                sigma_x_local,
                sigma_y_local,
            )
        })
        .collect();

    // -----------------------------
    // 5. Prepare filenames
    // -----------------------------
    let filenames: Vec<String> = dataset.iter().map(|s| io::filename_only(&s.path)).collect();

    // -----------------------------
    // 6. Save output
    // -----------------------------
    io::save_images_to_csv(&filenames, &images, &args.output)?;

    println!("✔ Saved to {:?}", args.output);

    Ok(())
}
