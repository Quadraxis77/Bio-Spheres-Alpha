//! Loads the two embedded GLB 2.0 assets, applying the full node hierarchy.
use glam::{Mat4, Quat, Vec3};
use serde_json::Value;
pub const LEFT: &[u8] = include_bytes!("../../assets/vr/touch-plus/left.glb");
pub const RIGHT: &[u8] = include_bytes!("../../assets/vr/touch-plus/right.glb");
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}
pub struct Model {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub png: Vec<u8>,
}
fn number(v: &Value) -> usize {
    v.as_u64().expect("GLB integer") as usize
}
fn array<const N: usize>(v: &Value, default: [f32; N]) -> [f32; N] {
    if v.is_null() {
        return default;
    }
    std::array::from_fn(|i| v[i].as_f64().expect("GLB number") as f32)
}
struct Glb<'a> {
    json: Value,
    bin: &'a [u8],
}
impl Glb<'_> {
    fn view(&self, index: usize) -> &[u8] {
        let view = &self.json["bufferViews"][index];
        let start = view["byteOffset"].as_u64().unwrap_or(0) as usize;
        &self.bin[start..start + number(&view["byteLength"])]
    }
    fn element(&self, index: usize, row: usize, size: usize) -> &[u8] {
        let a = &self.json["accessors"][index];
        assert!(row < number(&a["count"]));
        let view_index = number(&a["bufferView"]);
        let stride = self.json["bufferViews"][view_index]["byteStride"]
            .as_u64()
            .unwrap_or(size as u64) as usize;
        let offset = a["byteOffset"].as_u64().unwrap_or(0) as usize + row * stride;
        &self.view(view_index)[offset..offset + size]
    }
    fn floats<const N: usize>(&self, index: usize, row: usize) -> [f32; N] {
        assert_eq!(
            number(&self.json["accessors"][index]["componentType"]),
            5126
        );
        let bytes = self.element(index, row, N * 4);
        std::array::from_fn(|i| f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()))
    }
    fn node(&self, index: usize, parent: Mat4, model: &mut Model) {
        let node = &self.json["nodes"][index];
        let local = if node["matrix"].is_array() {
            Mat4::from_cols_array(&array(&node["matrix"], Mat4::IDENTITY.to_cols_array()))
        } else {
            Mat4::from_scale_rotation_translation(
                Vec3::from(array(&node["scale"], [1.0; 3])),
                Quat::from_array(array(&node["rotation"], [0.0, 0.0, 0.0, 1.0])),
                Vec3::from(array(&node["translation"], [0.0; 3])),
            )
        };
        let transform = parent * local;
        if let Some(mesh) = node["mesh"].as_u64() {
            for primitive in self.json["meshes"][mesh as usize]["primitives"]
                .as_array()
                .unwrap()
            {
                assert_eq!(primitive["mode"].as_u64().unwrap_or(4), 4);
                let attributes = &primitive["attributes"];
                let p = number(&attributes["POSITION"]);
                let n = number(&attributes["NORMAL"]);
                let uv = number(&attributes["TEXCOORD_0"]);
                let count = number(&self.json["accessors"][p]["count"]);
                let base = model.vertices.len() as u32;
                let normal_matrix = transform.inverse().transpose();
                for row in 0..count {
                    model.vertices.push(Vertex {
                        position: transform
                            .transform_point3(Vec3::from(self.floats::<3>(p, row)))
                            .to_array(),
                        normal: normal_matrix
                            .transform_vector3(Vec3::from(self.floats::<3>(n, row)))
                            .normalize()
                            .to_array(),
                        uv: self.floats(uv, row),
                    });
                }
                let accessor = number(&primitive["indices"]);
                let component = number(&self.json["accessors"][accessor]["componentType"]);
                for row in 0..number(&self.json["accessors"][accessor]["count"]) {
                    let value = match component {
                        5123 => {
                            u16::from_le_bytes(self.element(accessor, row, 2).try_into().unwrap())
                                as u32
                        }
                        5125 => {
                            u32::from_le_bytes(self.element(accessor, row, 4).try_into().unwrap())
                        }
                        _ => panic!("Unsupported embedded GLB index type"),
                    };
                    assert!(value < count as u32);
                    model.indices.push(base + value);
                }
            }
        }
        if let Some(children) = node["children"].as_array() {
            for child in children {
                self.node(number(child), transform, model);
            }
        }
    }
}
pub fn load(bytes: &[u8]) -> Model {
    let word = |offset| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    assert_eq!(word(0), 0x46546c67);
    assert_eq!(word(4), 2);
    assert_eq!(word(8) as usize, bytes.len());
    assert_eq!(word(16), 0x4e4f534a);
    let end = 20 + word(12) as usize;
    assert_eq!(word(end + 4), 0x004e4942);
    let glb = Glb {
        json: serde_json::from_slice(&bytes[20..end]).expect("Embedded GLB JSON"),
        bin: &bytes[end + 8..end + 8 + word(end) as usize],
    };
    let texture_index =
        number(&glb.json["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]["index"]);
    let image_index = number(&glb.json["textures"][texture_index]["source"]);
    assert_eq!(glb.json["images"][image_index]["mimeType"], "image/png");
    let mut model = Model {
        vertices: Vec::new(),
        indices: Vec::new(),
        png: glb
            .view(number(&glb.json["images"][image_index]["bufferView"]))
            .to_vec(),
    };
    let scene = glb.json["scene"].as_u64().unwrap_or(0) as usize;
    for node in glb.json["scenes"][scene]["nodes"].as_array().unwrap() {
        glb.node(number(node), Mat4::IDENTITY, &mut model);
    }
    model
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_assets_have_textures_buttons_and_physical_meter_dimensions() {
        for bytes in [LEFT, RIGHT] {
            let model = load(bytes);
            assert!(model.vertices.len() > 3000 && model.indices.len() > 12000);
            let mut min = Vec3::splat(f32::INFINITY);
            let mut max = Vec3::splat(f32::NEG_INFINITY);
            for vertex in &model.vertices {
                let p = Vec3::from(vertex.position);
                min = min.min(p);
                max = max.max(p);
                assert!(Vec3::from(vertex.normal).is_normalized());
            }
            let longest = (max - min).max_element();
            assert!(
                (0.10..0.20).contains(&longest),
                "Controller must be life size: {longest}"
            );
            assert!(image::load_from_memory(&model.png).unwrap().width() >= 512);
        }
    }
}
