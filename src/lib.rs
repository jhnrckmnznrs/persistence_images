pub mod dataset;
pub mod diagram;
pub mod image;
pub mod io;

// -----------------------------
// Public API (what users see)
// -----------------------------

pub use dataset::{SpacingDiagram, load_dataset, to_scaled_diagrams};

pub use diagram::{Diagram, read_diagram};

pub use image::{Bounds, compute_bounds, persistence_image};
