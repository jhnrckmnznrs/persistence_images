//! Exact (in real arithmetic) factorization for weighted pixel-integrated images.
//!
//! For distinct births beta_u and persistences pi_v, the image is A C D:
//! A[i,u] = integral over x pixel of N(x;beta_u,sx),
//! D[v,j] = min(1,pi_v/weight_scale) * integral over y pixel N(y;pi_v,sy),
//! C[u,v] = multiplicity. Cached bases are reused for matching spacing groups.
//! Floating-point accumulation may differ from the direct reference.
use crate::diagram::Diagram;
use crate::image::Bounds;
use crate::optimized::Engine;
use crate::stable::{cdf_interval, stable_persistence_image};
use rayon::prelude::*;
use std::collections::BTreeMap;

const MAX_COUNT_CELLS:usize=2_000_000;
const MAX_BASIS_ELEMENTS:usize=16_000_000;

fn unique(mut values:Vec<f64>)->Vec<f64>{
    values.sort_by(f64::total_cmp);
    values.dedup_by(|a,b|a.total_cmp(b).is_eq());
    values
}
fn validate(diagrams:&[&Diagram],bounds:Bounds,nx:usize,ny:usize,
            sx:f64,sy:f64,weight_scale:f64)->Result<(),String>{
    if nx==0 || ny==0 || nx.checked_mul(ny).is_none_or(|n|n>30_000_000) {
        return Err("Invalid stable image dimensions".into());
    }
    if !(bounds.x_min.is_finite()&&bounds.x_max.is_finite()&&bounds.x_min<bounds.x_max
       &&bounds.y_min.is_finite()&&bounds.y_max.is_finite()&&bounds.y_min<bounds.y_max
       &&sx.is_finite()&&sx>0.0&&sy.is_finite()&&sy>0.0
       &&weight_scale.is_finite()&&weight_scale>0.0){
        return Err("Stable basis needs finite nondegenerate bounds and positive scales".into());
    }
    if diagrams.iter().any(|d|d.iter().any(|&(b,p)|!b.is_finite()||!p.is_finite()||p<0.0)){
        return Err("Stable basis requires finite births and nonnegative persistence".into());
    }
    Ok(())
}

/// Reusable weighted Gaussian pixel-integral bases, shared across diagrams
/// with identical bounds, bandwidths, resolution, and persistence weight.
pub struct StableCdfBasis {
    births:Vec<f64>,persistences:Vec<f64>,
    a:Vec<f64>,d:Vec<f64>,nx:usize,ny:usize,
}
impl StableCdfBasis {
    pub fn from_diagrams(diagrams:&[&Diagram],bounds:Bounds,nx:usize,ny:usize,
                         sx:f64,sy:f64,weight_scale:f64)->Result<Self,String>{
        validate(diagrams,bounds,nx,ny,sx,sy,weight_scale)?;
        // Zero-persistence classes have weight zero and can be discarded.
        let births=unique(diagrams.iter().flat_map(|d|d.iter()
            .filter(|&&(_,p)|p>0.0).map(|&(b,_)|b)).collect());
        let persistences=unique(diagrams.iter().flat_map(|d|d.iter()
            .filter(|&&(_,p)|p>0.0).map(|&(_,p)|p)).collect());
        let (nb,np)=(births.len(),persistences.len());
        if nb.checked_mul(np).is_none_or(|x|x>MAX_COUNT_CELLS)
          || nx.checked_mul(nb).and_then(|x|ny.checked_mul(np)
                .and_then(|y|x.checked_add(y)))
                .is_none_or(|x|x>MAX_BASIS_ELEMENTS){
            return Err("Stable CDF basis exceeds memory limits".into());
        }
        let hx=(bounds.x_max-bounds.x_min)/nx as f64;
        let hy=(bounds.y_max-bounds.y_min)/ny as f64;
        if !hx.is_finite()||hx<=0.0||!hy.is_finite()||hy<=0.0 {
            return Err("Stable basis pixel spacing is invalid".into());
        }
        let mut a=vec![0.0;nx*nb];
        let mut d=vec![0.0;np*ny];
        for i in 0..nx {
            let left=bounds.x_min+(i as f64)*hx;
            let right=if i+1==nx {bounds.x_max}
                      else {bounds.x_min+((i+1) as f64)*hx};
            for (u,&b) in births.iter().enumerate(){
                a[i*nb+u]=cdf_interval(left,right,b,sx);
            }
        }
        for (v,&p) in persistences.iter().enumerate(){
            let weight=(p/weight_scale).min(1.0);
            for j in 0..ny {
                let left=bounds.y_min+(j as f64)*hy;
                let right=if j+1==ny {bounds.y_max}
                          else {bounds.y_min+((j+1) as f64)*hy};
                d[v*ny+j]=weight*cdf_interval(left,right,p,sy);
            }
        }
        Ok(Self{births,persistences,a,d,nx,ny})
    }
    pub fn coordinate_counts(&self)->(usize,usize){
        (self.births.len(),self.persistences.len())
    }
    pub fn transform(&self,diagram:&Diagram)->Result<Vec<f64>,String>{
        let (nb,np)=self.coordinate_counts();
        let mut out=vec![0.0;self.nx*self.ny];
        if nb==0 || np==0 {
            if diagram.iter().any(|&(_,p)|p>0.0){return Err("Point absent from CDF basis".into());}
            return Ok(out);
        }
        let mut counts=vec![0.0;nb*np];
        for &(b,p) in diagram{
            if !b.is_finite()||!p.is_finite()||p<0.0 {
                return Err("Nonfinite or negative diagram coordinate".into());
            }
            if p==0.0 {continue;}
            let u=self.births.binary_search_by(|x|x.total_cmp(&b))
                .map_err(|_|"Birth coordinate not in CDF basis")?;
            let v=self.persistences.binary_search_by(|x|x.total_cmp(&p))
                .map_err(|_|"Persistence coordinate not in CDF basis")?;
            counts[u*np+v]+=1.0;
        }
        let (nx,ny)=(self.nx,self.ny);
        // Choose the cheaper association (A C) D or A (C D).
        let left=(nx as u128)*(nb as u128)*(np as u128)
                  +(nx as u128)*(np as u128)*(ny as u128);
        let right=(nb as u128)*(np as u128)*(ny as u128)
                  +(nx as u128)*(nb as u128)*(ny as u128);
        if left<=right {
            let mut tmp=vec![0.0;nx*np];
            for u in 0..nb {for v in 0..np {
                let c=counts[u*np+v];
                if c==0.0 {continue;}
                for i in 0..nx {tmp[i*np+v]+=c*self.a[i*nb+u];}
            }}
            for i in 0..nx {for v in 0..np {
                let c=tmp[i*np+v];if c==0.0 {continue;}
                for j in 0..ny {out[i*ny+j]+=c*self.d[v*ny+j];}
            }}
        } else {
            let mut tmp=vec![0.0;nb*ny];
            for u in 0..nb {for v in 0..np {
                let c=counts[u*np+v];if c==0.0 {continue;}
                for j in 0..ny {tmp[u*ny+j]+=c*self.d[v*ny+j];}
            }}
            for i in 0..nx {for u in 0..nb {
                let c=self.a[i*nb+u]; if c==0.0 {continue;}
                for j in 0..ny {out[i*ny+j]+=c*tmp[u*ny+j];}
            }}
        }
        Ok(out)
    }
}
fn candidate(ds:&[&Diagram],nx:usize,ny:usize)->bool{
    if ds.len()<2||nx*ny<256 {return false;}
    let total:usize=ds.iter().map(|d|d.len()).sum();
    if total<512 {return false;}
    let sample:Vec<(f64,f64)>=ds.iter().flat_map(|d|d.iter()
        .filter(|&&(_,p)|p>0.0).take(512).copied()).take(2048).collect();
    if sample.is_empty(){return false;}
    let nb=unique(sample.iter().map(|&(b,_)|b).collect()).len();
    let np=unique(sample.iter().map(|&(_,p)|p).collect()).len();
    nb.saturating_mul(np)<=8*sample.len()
}
fn economical(b:&StableCdfBasis,ds:&[&Diagram],nx:usize,ny:usize)->bool {
    let (nb,np)=b.coordinate_counts();
    let n:usize=ds.iter().map(|d|d.len()).sum();
    let groups=ds.len();
    let shared=12_u128*((nx*nb+ny*np) as u128)
        +8_u128*(n as u128)
        +(groups as u128)*((nb as u128)*(np as u128)*(nx.min(ny) as u128)
           +(nx as u128)*(ny as u128)*(nb.min(np) as u128));
    let dense=(n as u128)*(nx as u128*ny as u128+4*(nx+ny) as u128);
    shared*10 < dense*6
}
/// Exact-integral stable batch with cost-guided reuse. Engine::Shared forces
/// reuse; Direct and Separable retain pointwise integration; Auto may reuse.
pub fn stable_batch_images_with_engine(
    diagrams:&[Diagram],spacings:&[f64],bounds:Bounds,nx:usize,ny:usize,
    sx:f64,sy:f64,weight_scale:f64,engine:Engine)->Result<Vec<Vec<f64>>,String>{
    if !matches!(engine,Engine::Auto|Engine::Direct|Engine::Separable|Engine::Shared){
        return Err("Stable images support auto, direct, separable and shared".into());
    }
    if diagrams.len()!=spacings.len() {return Err("spacings length mismatch".into());}
    let mut groups:BTreeMap<u64,Vec<usize>>=BTreeMap::new();
    for (i,&s) in spacings.iter().enumerate() {
        if !s.is_finite()||s<=0.0||!(sx*s).is_finite()||!(sy*s).is_finite()||
            sx*s<=0.0||sy*s<=0.0 {
            return Err("Scaled bandwidths must be finite and positive".into());
        }
        groups.entry(s.to_bits()).or_default().push(i);
    }
    let mut out=vec![Vec::new();diagrams.len()];
    for idx in groups.values(){
        let ds:Vec<&Diagram>=idx.iter().map(|&i|&diagrams[i]).collect();
        let spacing=spacings[idx[0]];
        let (local_sx,local_sy)=(sx*spacing,sy*spacing);
        if engine==Engine::Shared || (engine==Engine::Auto&&candidate(&ds,nx,ny)){
            match StableCdfBasis::from_diagrams(&ds,bounds,nx,ny,local_sx,local_sy,weight_scale){
                Ok(basis)=>{
                    if engine==Engine::Shared || economical(&basis,&ds,nx,ny) {
                        for &i in idx {out[i]=basis.transform(&diagrams[i])?;}
                        continue;
                    }
                }
                Err(e) if engine==Engine::Shared=>return Err(e),
                Err(_)=>{},
            }
        }
        let computed:Vec<Result<Vec<f64>,String>>=idx.par_iter().map(|&i|
            stable_persistence_image(&diagrams[i],bounds,nx,ny,
                local_sx,local_sy,weight_scale)).collect();
        for (&i,v) in idx.iter().zip(computed){out[i]=v?;}
    }
    Ok(out)
}

#[cfg(test)]
mod tests{
    use super::*;
    fn close(a:&[f64],b:&[f64],tol:f64){
        assert_eq!(a.len(),b.len());
        for (&x,&y) in a.iter().zip(b) {
            assert!((x-y).abs()<=tol*(1.0+x.abs()),
                "CDF factorization difference: {x} vs {y}");
        }
    }
    #[test]
    fn non_square_anisotropic_weight_and_diagonal(){
        let b=Bounds{x_min:-2.0,x_max:3.0,y_min:-1.0,y_max:4.0};
        let d=vec![(0.2,0.1),(0.2,0.1),(0.8,1.2),(1.1,0.0),(0.8,1.2)];
        let e=vec![(0.2,1.2),(0.8,0.1),(0.8,0.0)];
        let basis=StableCdfBasis::from_diagrams(&[&d,&e],b,23,37,0.35,0.72,0.4).unwrap();
        assert_eq!(basis.coordinate_counts(),(2,2));
        close(&basis.transform(&d).unwrap(),&stable_persistence_image(&d,b,23,37,0.35,0.72,0.4).unwrap(),1e-12);
        close(&basis.transform(&e).unwrap(),&stable_persistence_image(&e,b,23,37,0.35,0.72,0.4).unwrap(),1e-12);
        assert!(basis.transform(&vec![(0.7,2.0)]).is_err());
    }
    #[test]
    fn mixed_spacing_groups_and_empty(){
        let b=Bounds{x_min:-1.0,x_max:3.0,y_min:0.0,y_max:4.0};
        let a=vec![(0.2,0.5);100];
        let c=vec![(1.1,0.1);95];
        let d=vec![(0.2,0.0);50];
        let diagrams=vec![a,c,d,vec![]];
        let spacing=vec![1.0,2.0,1.0,2.0];
        for engine in [Engine::Shared,Engine::Auto]{
            let result=stable_batch_images_with_engine(&diagrams,&spacing,b,32,25,
                0.4,0.3,1.0,engine).unwrap();
            for (i,d) in diagrams.iter().enumerate() {
                close(&result[i],&stable_persistence_image(d,b,32,25,0.4*spacing[i],
                        0.3*spacing[i],1.0).unwrap(),1e-12);
            }
        }
    }
    #[test]
    fn invalid(){
        let b=Bounds{x_min:0.0,x_max:1.0,y_min:0.0,y_max:1.0};
        assert!(StableCdfBasis::from_diagrams(&[&vec![(0.1,-0.1)]],b,10,10,0.2,0.2,1.0).is_err());
        assert!(stable_batch_images_with_engine(&[vec![]],&[1.0],b,10,10,0.2,0.2,1.0,
            Engine::Sparse).is_err());
    }
}
