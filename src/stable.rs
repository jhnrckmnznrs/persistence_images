//! Probability-normalized, weighted, pixel-integrated persistence images.
//!
//! This is a separate representation from the legacy unweighted pixel-center
//! evaluator.  Coordinates are (birth,persistence), with persistence >= 0.
//! Fixed grid, bandwidths and ramp scale are required for the W1 stability
//! statement; never normalize the output vector by its own sum.
use crate::diagram::Diagram;
use crate::image::Bounds;
use rayon::prelude::*;
use std::f64::consts::SQRT_2;

/// Gaussian mass within [left,right] for a normal law centered at mean.
/// Choose a complementary-CDF formula in positive tails to avoid 1-1 loss.
fn cdf_interval(left:f64,right:f64,mean:f64,sigma:f64)->f64 {
    let lo=(left-mean)/(sigma*SQRT_2);
    let hi=(right-mean)/(sigma*SQRT_2);
    let answer=if lo >= 0.0 {
        0.5*(libm::erfc(lo)-libm::erfc(hi))
    } else if hi <= 0.0 {
        0.5*(libm::erfc(-hi)-libm::erfc(-lo))
    } else {
        0.5*(libm::erf(hi)-libm::erf(lo))
    };
    answer.max(0.0)
}

fn validate(diagram:&Diagram,bounds:Bounds,nx:usize,ny:usize,
            sx:f64,sy:f64,weight_scale:f64)->Result<(),String>{
    let n=nx.checked_mul(ny).ok_or("Pixel count overflows usize")?;
    if nx==0 || ny==0 || n > 30_000_000 {
        return Err("Grid sizes must be positive; at most 30 million pixels".into());
    }
    if !(sx.is_finite()&&sx>0.0&&sy.is_finite()&&sy>0.0
        &&weight_scale.is_finite()&&weight_scale>0.0) {
        return Err("Bandwidths and weight_scale must be finite and positive".into());
    }
    if !(bounds.x_min.is_finite()&&bounds.x_max.is_finite()
        &&bounds.y_min.is_finite()&&bounds.y_max.is_finite()
        &&bounds.x_min<bounds.x_max&&bounds.y_min<bounds.y_max) {
        return Err("Stable images require finite, nondegenerate fixed bounds".into());
    }
    if diagram.iter().any(|&(b,p)| !b.is_finite()||!p.is_finite()||p<0.0) {
        return Err("Stable images require finite birth and nonnegative finite persistence".into());
    }
    Ok(())
}

/// Exact-in-real-arithmetic pixel integrals of a weighted normalized
/// anisotropic Gaussian. The output uses index i*ny+j.
///
/// The bounded ramp weight is min(1,persistence/weight_scale), zero on the
/// birth-death diagonal. Each Gaussian density integrates to 1 on R²;
/// the resulting image is NOT normalized to sum to 1.
///
/// Numeric error comes from Gaussian CDF and floating-point accumulation.
pub fn stable_persistence_image(diagram:&Diagram,bounds:Bounds,nx:usize,ny:usize,
                                sx:f64,sy:f64,weight_scale:f64)
                                ->Result<Vec<f64>,String> {
    validate(diagram,bounds,nx,ny,sx,sy,weight_scale)?;
    let hx=(bounds.x_max-bounds.x_min)/nx as f64;
    let hy=(bounds.y_max-bounds.y_min)/ny as f64;
    if !hx.is_finite()||!hy.is_finite()||hx<=0.0||hy<=0.0 {
        return Err("Grid edge spacing must be positive and finite".into());
    }
    let mut result=vec![0.0;nx*ny];
    let mut mass_x=vec![0.0;nx];
    let mut mass_y=vec![0.0;ny];
    for &(birth,persistence) in diagram {
        let weight=(persistence/weight_scale).min(1.0);
        if weight==0.0 {continue;}
        for i in 0..nx {
            // Snap last edge to supplied bound; avoid drift from adding steps.
            let l=bounds.x_min+(i as f64)*hx;
            let r=if i+1==nx {bounds.x_max} else {bounds.x_min+((i+1) as f64)*hx};
            mass_x[i]=cdf_interval(l,r,birth,sx)*weight;
        }
        for j in 0..ny {
            let l=bounds.y_min+(j as f64)*hy;
            let r=if j+1==ny {bounds.y_max} else {bounds.y_min+((j+1) as f64)*hy};
            mass_y[j]=cdf_interval(l,r,persistence,sy);
        }
        for i in 0..nx {
            if mass_x[i]==0.0 {continue;}
            for j in 0..ny {
                result[i*ny+j]+=mass_x[i]*mass_y[j];
            }
        }
    }
    Ok(result)
}

/// Diagrams must already have voxel spacing absorbed into the coordinates.
/// As in the legacy pipeline, the spacings scale the base Gaussian bandwidths.
/// The ramp scale stays fixed in the scaled persistence coordinate units.
pub fn stable_batch_images(diagrams:&[Diagram],spacings:&[f64],bounds:Bounds,
                           nx:usize,ny:usize,sx:f64,sy:f64,weight_scale:f64)
                           ->Result<Vec<Vec<f64>>,String> {
    if diagrams.len()!=spacings.len() {return Err("spacings length mismatch".into());}
    let results:Vec<Result<Vec<f64>,String>>=diagrams.par_iter()
        .zip(spacings.par_iter())
        .map(|(d,&s)|{
            if !s.is_finite()||s<=0.0 {
                return Err("voxel spacing must be finite and positive".into());
            }
            stable_persistence_image(d,bounds,nx,ny,sx*s,sy*s,weight_scale)
        }).collect();
    results.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn box_()->Bounds {
        Bounds{x_min:-10.0,x_max:10.0,y_min:-10.0,y_max:10.0}
    }
    #[test]
    fn gaussian_mass_and_diagonal(){
        let b=box_();
        let empty=stable_persistence_image(&vec![],b,64,40,1.0,1.0,1.0).unwrap();
        assert!(empty.iter().all(|&a|a==0.0));
        let diagonal=stable_persistence_image(&vec![(0.0,0.0)],b,64,40,1.0,1.0,1.0).unwrap();
        assert_eq!(diagonal,empty);
        let output=stable_persistence_image(&vec![(0.0,1.0)],b,64,40,1.0,1.0,1.0).unwrap();
        let mass: f64=output.iter().sum();
        assert!((mass-1.0).abs()<1e-12, "normalized mass {mass}");
        let output2=stable_persistence_image(&vec![(0.0,0.2)],b,64,40,1.0,1.0,1.0).unwrap();
        let mass2:f64=output2.iter().sum();
        assert!((mass2-0.2).abs()<1e-12,"weighted mass {mass2}");
    }
    #[test]
    fn single_pixel_cdf_and_tail(){
        let b=Bounds{x_min:-1.0,x_max:1.0,y_min:-1.0,y_max:1.0};
        let v=stable_persistence_image(&vec![(0.0,1.0)],b,1,1,1.0,1.0,1.0).unwrap();
        let x=libm::erf(1.0/SQRT_2);
        let y=0.5*(libm::erf(0.0)-libm::erf(-2.0/SQRT_2));
        assert!((v[0]-x*y).abs()<1e-14);
        let tail=stable_persistence_image(&vec![(14.0,1.0)],b,1,1,1.0,1.0,1.0).unwrap();
        assert!(tail[0]>=0.0 && tail[0]<1e-32);
    }
    #[test]
    fn stable_under_perturbations_and_batch(){
        let b=Bounds{x_min:0.0,x_max:3.0,y_min:0.0,y_max:3.0};
        let d=vec![(0.5,0.7),(1.4,0.2)];
        let a=stable_persistence_image(&d,b,30,24,0.4,0.6,1.0).unwrap();
        let q=vec![(0.50001,0.70002),(1.4,0.2)];
        let c=stable_persistence_image(&q,b,30,24,0.4,0.6,1.0).unwrap();
        let l1:f64=a.iter().zip(c.iter()).map(|(x,y)|(x-y).abs()).sum();
        let constant=2.0+(2.0/std::f64::consts::PI).sqrt()*(1.0/0.4+2.0/0.6);
        assert!(l1<=constant*0.00003);
        let out=stable_batch_images(&[d.clone(),q.clone()],&[1.0,2.0],b,30,24,
            0.4,0.6,1.0).unwrap();
        assert_eq!(out[0],a);
        assert_eq!(out.len(),2);
        assert_eq!(out[1],
            stable_persistence_image(&q,b,30,24,0.8,1.2,1.0).unwrap());
    }
    #[test]
    fn invalid_inputs(){
        let b=box_();
        assert!(stable_persistence_image(&vec![(0.0,-0.01)],b,2,3,1.0,1.0,1.0).is_err());
        assert!(stable_persistence_image(&vec![(0.0,f64::INFINITY)],b,2,3,1.0,1.0,1.0).is_err());
        assert!(stable_persistence_image(&vec![],b,0,3,1.0,1.0,1.0).is_err());
        assert!(stable_persistence_image(&vec![],b,2,3,1.0,1.0,0.0).is_err());
        assert!(stable_persistence_image(&vec![],Bounds{x_min:0.0,x_max:0.0,..b},2,3,1.0,1.0,1.0).is_err());
    }
}
