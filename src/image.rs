use crate::diagram::Diagram;

/// Axis-aligned bounding box for persistence diagrams.
#[derive(Debug, Clone, Copy)]
pub struct Bounds {
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
}

/// Computes an anisotropic Gaussian kernel value.
fn gaussian_aniso(dx: f64, dy: f64, sigma_x: f64, sigma_y: f64) -> f64 {
    let x_term = dx * dx / (2.0 * sigma_x * sigma_x);
    let y_term = dy * dy / (2.0 * sigma_y * sigma_y);
    (-(x_term + y_term)).exp()
}

/// Computes a persistence image from a diagram.
///
/// # Arguments
/// - `diagram`: (birth, persistence) pairs
/// - `bounds`: global bounding box
/// - `nx`, `ny`: resolution
/// - `sigma_x`, `sigma_y`: Gaussian bandwidths
///
/// # Returns
/// Flattened image (row-major)
pub fn persistence_image(
    diagram: &Diagram,
    bounds: Bounds,
    nx: usize,
    ny: usize,
    sigma_x: f64,
    sigma_y: f64,
) -> Vec<f64> {
    let mut image = vec![0.0; nx * ny];

    let dx = (bounds.x_max - bounds.x_min) / nx as f64;
    let dy = (bounds.y_max - bounds.y_min) / ny as f64;

    let xs: Vec<f64> = (0..nx)
        .map(|i| bounds.x_min + (i as f64 + 0.5) * dx)
        .collect();

    let ys: Vec<f64> = (0..ny)
        .map(|j| bounds.y_min + (j as f64 + 0.5) * dy)
        .collect();

    // O(nx * ny * |diagram|)
    for (i, &x) in xs.iter().enumerate() {
        for (j, &y) in ys.iter().enumerate() {
            let mut value = 0.0;

            for &(birth, persistence) in diagram {
                let dx = x - birth;
                let dy = y - persistence;

                value += gaussian_aniso(dx, dy, sigma_x, sigma_y);
            }

            image[i * ny + j] = value;
        }
    }

    image
}

/// Computes bounds across multiple diagrams.
///
/// Returns `None` if all diagrams are empty.
pub fn compute_bounds(diagrams: &[Diagram]) -> Option<Bounds> {
    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;

    let mut found_any = false;

    for diagram in diagrams {
        for &(birth, persistence) in diagram {
            found_any = true;

            x_min = x_min.min(birth);
            x_max = x_max.max(birth);
            y_min = y_min.min(persistence);
            y_max = y_max.max(persistence);
        }
    }

    if !found_any {
        return None;
    }

    Some(Bounds {
        x_min,
        x_max,
        y_min,
        y_max,
    })
}
