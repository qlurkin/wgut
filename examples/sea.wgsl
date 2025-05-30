// based on https://www.shadertoy.com/view/Ms2SD1

const NUM_STEPS = 8;
const PI = 3.141592;
const EPSILON = 0.001;
const ITER_GEOMETRY = 3;
const ITER_FRAGMENT = 5;
const SEA_HEIGHT = 0.6;
const SEA_CHOPPY = 4.0;
const SEA_SPEED = 0.8;
const SEA_FREQ = 0.16;
const SEA_BASE = vec3<f32>(0.0, 0.09, 0.18);
const SEA_WATER_COLOR = vec3<f32>(0.48, 0.54, 0.36);

fn hash(p: vec2<f32>) -> f32 {

    let h = dot(p, vec2<f32>(1.0, 113.0));
    return fract(sin(h) * 43758.5453123);
}

// Perlin noise
fn noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let mix1 = mix(hash(i + vec2<f32>(0.0, 0.0)), hash(i + vec2<f32>(1.0, 0.0)), u.x);
    let mix2 = mix(hash(i + vec2<f32>(0.0, 1.0)), hash(i + vec2<f32>(1.0, 1.0)), u.x);
    let mix3 = mix(mix1, mix2, u.y);
    return -1.0 + 2.0 * mix3;
}

// lighting
fn diffuse(n: vec3<f32>, l: vec3<f32>, p: f32) -> f32 {
    return pow(dot(n, l) * 0.4 + 0.6, p);
}

fn specular(n: vec3<f32>, l: vec3<f32>, e: vec3<f32>, s: f32) -> f32 {
    let nrm = (s + 8.0) / (PI * 8.0);
    return pow(max(dot(reflect(e, n), l), 0.0), s) * nrm;
}

// sky
fn getSkyColor(_e: vec3<f32>) -> vec3<f32> {
    var e = _e;
    e.y = (max(e.y, 0.0) * 0.8 + 0.2) * 0.8;
    return vec3<f32>(pow(1.0 - e.y, 2.0), 1.0 - e.y, 0.6 + (1.0 - e.y) * 0.4) * 1.1;
}

// sea
fn sea_octave(_uv: vec2<f32>, choppy: f32) -> f32 {
    let uv = _uv + noise(_uv);
    var wv = 1.0 - abs(sin(uv));
    let swv = abs(cos(uv));
    wv = mix(wv, swv, wv);
    return pow(1.0 - pow(wv.x * wv.y, 0.65), choppy);
}

fn _map(p: vec3<f32>, iter: i32) -> f32 {
    var freq = SEA_FREQ;
    var amp = SEA_HEIGHT;
    var choppy = SEA_CHOPPY;
    let sea_time = 1.0 + i_time * SEA_SPEED;
    var uv = p.xz;
    uv.x *= 0.75;
    var d = 0.0;
    var h = 0.0;
    for (var i = 0; i < iter; i += 1) {
        d = sea_octave((uv + sea_time) * freq, choppy);
        d += sea_octave((uv - sea_time) * freq, choppy);
        h += d * amp;
        uv *= mat2x2<f32>(1.6, 1.2, -1.2, 1.6);
        freq *= 1.9;
        amp *= 0.22;
        choppy = mix(choppy, 1.0, 0.2);
    }
    return p.y - h;
}

fn map(p: vec3<f32>) -> f32 {
    return _map(p, ITER_GEOMETRY);
}

fn map_detailed(p: vec3<f32>) -> f32 {
    return _map(p, ITER_FRAGMENT);
}

fn getSeaColor(p: vec3<f32>, n: vec3<f32>, l: vec3<f32>, eye: vec3<f32>, dist: vec3<f32>) -> vec3<f32> {
    var fresnel = clamp(1.0 - dot(n, -eye), 0.0, 1.0);
    fresnel = pow(fresnel, 3.0) * 0.5;
    let reflected = getSkyColor(reflect(eye, n));
    let refracted = SEA_BASE + diffuse(n, l, 80.0) * SEA_WATER_COLOR * 0.12;
    var color = mix(refracted, reflected, fresnel);
    let atten = max(1.0 - dot(dist, dist) * 0.001, 0.0);
    color += SEA_WATER_COLOR * (p.y - SEA_HEIGHT) * 0.18 * atten;
    color += vec3<f32>(specular(n, l, eye, 60.0));
    return color;
}

// tracing
fn getNormal(p: vec3<f32>, eps: f32) -> vec3<f32> {
    var n: vec3<f32>;
    n.y = map_detailed(p);
    n.x = map_detailed(vec3<f32>(p.x + eps, p.y, p.z)) - n.y;
    n.z = map_detailed(vec3<f32>(p.x, p.y, p.z + eps)) - n.y;
    n.y = eps;
    return normalize(n);
}

fn heightMapTracing(ori: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    var tm = 0.0;
    var tx = 1000.0;
    var hx = map(ori + dir * tx);
    var p: vec3<f32>;
    if hx > 0.0 {
        p = ori + dir * tx;
        return p;
    }
    var hm = map(ori + dir * tm);
    var tmid = 0.0;
    for (var i = 0; i < NUM_STEPS; i += 1) {
        tmid = mix(tm, tx, hm / (hm - hx));
        p = ori + dir * tmid;
        let hmid = map(p);
        if hmid < 0.0 {
            tx = tmid;
            hx = hmid;
        } else {
            tm = tmid;
            hm = hmid;
        }
    }
    return p;
}

fn getPixel(coord: vec2<f32>, time: f32) -> vec3<f32> {
    var uv = coord / i_resolution.xy;
    uv = uv * 2.0 - 1.0;
    uv.x *= i_resolution.x / i_resolution.y;
    // ray
    let ori = vec3<f32>(0.0, 3.5, time * 5.0);
    let dir = normalize(vec3<f32>(uv.xy, -2.0));
    // tracing
    var p = heightMapTracing(ori, dir);
    let dist = p - ori;
    let n = getNormal(p, dot(dist, dist) * (0.1 / i_resolution.x));
    let light = normalize(vec3<f32>(0.0, 1.0, 0.8));
    // color
    return mix(
        getSkyColor(dir),
        getSeaColor(p, n, light, dir, dist),
        pow(smoothstep(0.0, -0.02, dir.y), 0.2)
    );
}

fn main_image(frag_coord: vec2<f32>) -> vec4<f32> {
    let time = i_time * 0.3;
    let color = getPixel(frag_coord, time);
    return vec4<f32>(pow(color, vec3<f32>(0.65)), 1.0);
}

