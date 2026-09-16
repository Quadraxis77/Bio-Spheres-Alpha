//! Optional hardware cave occlusion. Geometry is uploaded only on cave edits.
use wgpu::util::DeviceExt;

pub struct CaveRayTracing {
    pending: Option<Vec<u32>>,
    pub tlas: Option<wgpu::Tlas>,
}
impl CaveRayTracing {
    pub fn new() -> Self {
        Self {
            pending: None,
            tlas: None,
        }
    }
    pub fn set_solid_mask(&mut self, mask: &[u32]) {
        assert_eq!(mask.len(), 128usize.pow(3));
        self.pending = Some(mask.to_vec());
        // Never trace against the previous cave after an edit.
        self.tlas = None;
    }
    pub fn prepare(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        let Some(mask) = self.pending.take() else {
            return;
        };
        let limit = device
            .limits()
            .max_blas_primitive_count
            .min(2_000_000)
            .min((device.limits().max_buffer_size / 36) as u32) as usize;
        let Some(mut vertices) = cave_triangles(&mask, 128, limit) else {
            log::warn!("Cave exceeds ray tracing geometry budget; using voxel occlusion");
            return;
        };
        // A degenerate triangle provides a valid empty BLAS without an occluder.
        if vertices.is_empty() {
            vertices.resize(3, [0.; 3]);
        }
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Luminocyte cave ray geometry"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::BLAS_INPUT,
        });
        let size = wgpu::BlasTriangleGeometrySizeDescriptor {
            vertex_format: wgpu::VertexFormat::Float32x3,
            vertex_count: vertices.len() as u32,
            index_format: None,
            index_count: None,
            flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
        };
        let blas = device.create_blas(
            &wgpu::CreateBlasDescriptor {
                label: Some("Luminocyte cave BLAS"),
                flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            },
            wgpu::BlasGeometrySizeDescriptors::Triangles {
                descriptors: vec![size.clone()],
            },
        );
        let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
            label: Some("Luminocyte cave TLAS"),
            max_instances: 1,
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        });
        // Geometry and rays are both in light-grid coordinates.
        tlas[0] = Some(wgpu::TlasInstance::new(
            &blas,
            [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.],
            0,
            0xff,
        ));
        encoder.build_acceleration_structures(
            &[wgpu::BlasBuildEntry {
                blas: &blas,
                geometry: wgpu::BlasGeometries::TriangleGeometries(vec![
                    wgpu::BlasTriangleGeometry {
                        size: &size,
                        vertex_buffer: &vertex_buffer,
                        first_vertex: 0,
                        vertex_stride: 12,
                        index_buffer: None,
                        first_index: None,
                        transform_buffer: None,
                        transform_buffer_offset: None,
                    },
                ]),
            }],
            [&tlas],
        );
        self.tlas = Some(tlas);
    }
}

/// Merge adjacent solid voxels along X. Each run is a closed opaque box.
/// Bounds match the physics mask exactly, including vent mouths and culled rock.
fn cave_triangles(mask: &[u32], res: usize, max_triangles: usize) -> Option<Vec<[f32; 3]>> {
    let mut vertices = Vec::new();
    for z in 0..res {
        for y in 0..res {
            let row = (z * res + y) * res;
            let mut x = 0;
            while x < res {
                if mask[row + x] == 0 {
                    x += 1;
                    continue;
                }
                let start = x;
                while x < res && mask[row + x] != 0 {
                    x += 1;
                }
                if vertices.len() / 3 + 12 > max_triangles {
                    return None;
                }
                let a = start as f32;
                let b = x as f32;
                let y = y as f32;
                let z = z as f32;
                let corners = [
                    [a, y, z],
                    [b, y, z],
                    [b, y + 1., z],
                    [a, y + 1., z],
                    [a, y, z + 1.],
                    [b, y, z + 1.],
                    [b, y + 1., z + 1.],
                    [a, y + 1., z + 1.],
                ];
                for face in [
                    [0, 1, 2, 3],
                    [4, 7, 6, 5],
                    [0, 4, 5, 1],
                    [3, 2, 6, 7],
                    [0, 3, 7, 4],
                    [1, 5, 6, 2],
                ] {
                    for i in [face[0], face[1], face[2], face[0], face[2], face[3]] {
                        vertices.push(corners[i]);
                    }
                }
            }
        }
    }
    Some(vertices)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mesh_merges_runs_preserves_gaps_and_obeys_budget() {
        let mask = [
            1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        let mesh = cave_triangles(&mask, 3, 24).unwrap();
        assert_eq!(mesh.len(), 72);
        assert!(mesh[..36]
            .iter()
            .all(|p| p[0] <= 2. && p[1] <= 1. && p[2] <= 1.));
        assert!(mesh[36..]
            .iter()
            .all(|p| (1. ..=2.).contains(&p[0]) && (1. ..=2.).contains(&p[1])));
        assert!(cave_triangles(&mask, 3, 23).is_none());
        assert!(cave_triangles(&[0; 27], 3, 1).unwrap().is_empty());
    }
}
