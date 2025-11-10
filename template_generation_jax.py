import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Complex


def xyyz_from_detector_geometry(
    pixel_size_m: float, detector_distance_m: float, n_pixels_x: int, n_pixels_y: int
) -> tuple[Float[Array, "2 max_pixels"], float]:
    """
    Calculate the x,y coordinates of each pixel on the detector plane
    and the z coordinate (detector distance).

    This function computes the spatial coordinates for each pixel on a 2D detector
    in a coordinate system where the detector center is at the origin, and the
    z-axis points from the sample towards the detector.

    Parameters
    ----------
    pixel_size_m : float
        Size of each detector pixel in meters (assumes square pixels)
    detector_distance_m : float
        Distance from sample to detector in meters
    n_pixels_x : int
        Number of pixels in the x-direction (horizontal)
    n_pixels_y : int
        Number of pixels in the y-direction (vertical)

    Returns
    -------
    xy : Float[Array, "2 max_pixels"]
        Array of shape (2, max(n_pixels_x, n_pixels_y)) containing x and y
        coordinates in meters. x coordinates are in xy[0,:] and y coordinates
        are in xy[1,:]. Coordinates are centered such that pixel (0,0)
        corresponds to the center of the detector.
    z : float
        The detector distance in meters (same as input detector_distance_m)

    Notes
    -----
    - The coordinate system places (0,0) at the detector center
    - Pixel coordinates are calculated as: (pixel_index - center_index) * pixel_size
    - The z-coordinate is constant for all pixels (flat detector assumption)
    - Used in conjunction with q_from_xyz() to calculate scattering vectors

    Examples
    --------
    >>> xy, z = xyyz_from_detector_geometry(200e-6, 1.2, 360, 360)
    >>> print(f"Detector spans from {xy[0,0]:.3f} to {xy[0,-1]:.3f} m in x")
    >>> print(f"Z-distance: {z} m")
    """
    x = (jnp.arange(n_pixels_x) - (n_pixels_x - 1) / 2) * pixel_size_m
    y = (jnp.arange(n_pixels_y) - (n_pixels_y - 1) / 2) * pixel_size_m
    z = detector_distance_m
    return jnp.stack((x, y)), z


xyyz_from_detector_geometry = jax.jit(
    xyyz_from_detector_geometry, static_argnames=("n_pixels_x", "n_pixels_y")
)


@jax.jit
def q_from_xyz(
    xy: Float[Array, "2 N"],
    z: float,
    wavelength_m: float,
) -> Float[Array, "3 N_x N_y"]:
    """
    Calculate the scattering vector components qx, qy, qz from the
    x,y,z coordinates and wavelength.

    This function converts real-space detector pixel coordinates to reciprocal
    space scattering vectors using the Ewald sphere geometry for small-angle
    X-ray scattering (SAXS).

    Parameters
    ----------
    xy : Float[Array, "2 N"]
        Array of shape (2, N) containing x and y coordinates in meters
        for detector pixels. xy[0,:] are x-coordinates, xy[1,:] are y-coordinates.
    z : float
        Detector distance from sample in meters (constant for all pixels)
    wavelength_m : float
        X-ray wavelength in meters

    Returns
    -------
    q_xyz : Float[Array, "3 N_x N_y"]
        Array of shape (3, N_x, N_y) containing scattering vector components
        in inverse meters. q_xyz[0] = qx, q_xyz[1] = qy, q_xyz[2] = qz.
        The scattering vectors follow the convention where the incident
        beam is along +z direction.

    Notes
    -----
    - Uses the small-angle approximation for SAXS geometry
    - The scattering vector magnitude is |q| = (4π/λ)sin(θ/2) ≈ 2πθ/λ for small θ
    - qz component includes the -1 term for forward scattering geometry
    - Coordinates are meshgridded internally to create 2D detector arrays

    Examples
    --------
    >>> xy, z = xyyz_from_detector_geometry(200e-6, 1.2, 360, 360)
    >>> q_vectors = q_from_xyz(xy, z, 0.207e-9)
    >>> print(f"Q-space shape: {q_vectors.shape}")
    """
    x, y = jnp.meshgrid(xy[0, :], xy[1, :])
    r = jnp.sqrt(x**2 + y**2 + z**2)
    k = 2 * jnp.pi / wavelength_m
    q_x, q_y, q_z = k * x / r, k * y / r, k * (z / r - 1)
    return jnp.stack((q_x, q_y, q_z))


@jax.jit
def get_rotation_matrix(
    rx_deg: float, ry_deg: float, rz_deg: float
) -> Float[Array, "3 3"]:
    """
    Get rotation matrix from Euler angles (degrees).

    Constructs a 3D rotation matrix using the ZYX Euler angle convention
    (also known as Tait-Bryan angles). The rotation is applied in the order:
    first around X-axis, then Y-axis, then Z-axis.

    Parameters
    ----------
    rx_deg : float
        Rotation angle around x-axis in degrees
    ry_deg : float
        Rotation angle around y-axis in degrees
    rz_deg : float
        Rotation angle around z-axis in degrees

    Returns
    -------
    R : Float[Array, "3 3"]
        3x3 rotation matrix. When applied to a vector v as R @ v,
        it rotates the vector by the specified Euler angles.

    Notes
    -----
    - Uses ZYX convention: R = Rz(rz) @ Ry(ry) @ Rx(rx)
    - Angles are converted from degrees to radians internally
    - The resulting matrix is orthogonal with determinant +1
    - Commonly used to rotate scattering vectors into particle frame

    Examples
    --------
    >>> R = get_rotation_matrix(30, 45, 60)
    >>> print(f"Rotation matrix shape: {R.shape}")
    >>> # Apply to scattering vectors
    >>> q_rotated = R @ q_vectors.reshape(3, -1)
    """
    rx_rad = jnp.deg2rad(rx_deg)
    ry_rad = jnp.deg2rad(ry_deg)
    rz_rad = jnp.deg2rad(rz_deg)

    ca, sa = jnp.cos(rx_rad), jnp.sin(rx_rad)
    cb, sb = jnp.cos(ry_rad), jnp.sin(ry_rad)
    cg, sg = jnp.cos(rz_rad), jnp.sin(rz_rad)

    # rotation matrix using ZYX convention
    Rx = jnp.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    Ry = jnp.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rz = jnp.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])

    return Rz @ Ry @ Rx


@jax.jit
def parallelepiped_form_factor(
    qx_p: Float[Array, "N_x N_y"],
    qy_p: Float[Array, "N_x N_y"],
    qz_p: Float[Array, "N_x N_y"],
    A: float,
    B: float,
    C: float,
    delta_rho: float = 1.0,
) -> Float[Array, "N_x N_y"]:
    """
    Calculate the analytical form factor for a homogeneous parallelepiped.

    Computes the scattering amplitude and intensity for a rectangular
    parallelepiped (box) aligned with the coordinate axes using the
    analytical solution for the Fourier transform.

    Parameters
    ----------
    qx_p, qy_p, qz_p : Float[Array, "N_x N_y"]
        Arrays of scattering vector components in particle frame (same shape).
        Should be in units of inverse meters (m⁻¹).
    A, B, C : float
        Side lengths of the parallelepiped along x, y, z axes in meters.
        A is x-dimension, B is y-dimension, C is z-dimension.
    delta_rho : float, optional
        Electron density contrast between particle and solvent in
        arbitrary units. Default is 1.0.

    Returns
    -------
    intensity : Float[Array, "N_x N_y"]
        Normalized scattering intensity |F(q)|² with same shape as input q arrays.
        Values are normalized by the maximum intensity (forward scattering).

    Notes
    -----
    - Uses the analytical form: F(q) = V * sinc(qx*A/2) * sinc(qy*B/2) * sinc(qz*C/2)
    - Where V = A*B*C is the particle volume
    - sinc(x) = sin(πx)/(πx) is the normalized sinc function
    - The intensity shows characteristic oscillations due to particle shape
    - Forward scattering (q=0) gives the total scattering power

    Examples
    --------
    >>> # 20x20x20 nm cube
    >>> intensity = parallelepiped_form_factor(qx, qy, qz, 20e-9, 20e-9, 20e-9)
    >>> # Rectangular particle
    >>> intensity = parallelepiped_form_factor(qx, qy, qz, 50e-9, 20e-9, 10e-9)
    """
    V = A * B * C
    argx = 0.5 * qx_p * A
    argy = 0.5 * qy_p * B
    argz = 0.5 * qz_p * C
    Fx = jnp.sinc(argx)
    Fy = jnp.sinc(argy)
    Fz = jnp.sinc(argz)
    F = jnp.nan_to_num(delta_rho * Fx * Fy * Fz, nan=1.0)  # *8/V
    return (F / F.max()) ** 2


@jax.jit
def parallelepiped_form_factor_complex(
    qx_p: Float[Array, "N_x N_y"],
    qy_p: Float[Array, "N_x N_y"],
    qz_p: Float[Array, "N_x N_y"],
    A: float,
    B: float,
    C: float,
    delta_rho: float = 1.0,
) -> Float[Array, "N_x N_y"]:
    """
    Calculate the analytical form factor for a homogeneous parallelepiped.

    Computes the scattering amplitude and intensity for a rectangular
    parallelepiped (box) aligned with the coordinate axes using the
    analytical solution for the Fourier transform.

    Parameters
    ----------
    qx_p, qy_p, qz_p : Float[Array, "N_x N_y"]
        Arrays of scattering vector components in particle frame (same shape).
        Should be in units of inverse meters (m⁻¹).
    A, B, C : float
        Side lengths of the parallelepiped along x, y, z axes in meters.
        A is x-dimension, B is y-dimension, C is z-dimension.
    delta_rho : float, optional
        Electron density contrast between particle and solvent in
        arbitrary units. Default is 1.0.

    Returns
    -------
    intensity : Float[Array, "N_x N_y"]
        Normalized scattering intensity |F(q)|² with same shape as input q arrays.
        Values are normalized by the maximum intensity (forward scattering).

    Notes
    -----
    - Uses the analytical form: F(q) = V * sinc(qx*A/2) * sinc(qy*B/2) * sinc(qz*C/2)
    - Where V = A*B*C is the particle volume
    - sinc(x) = sin(πx)/(πx) is the normalized sinc function
    - The intensity shows characteristic oscillations due to particle shape
    - Forward scattering (q=0) gives the total scattering power

    Examples
    --------
    >>> # 20x20x20 nm cube
    >>> intensity = parallelepiped_form_factor(qx, qy, qz, 20e-9, 20e-9, 20e-9)
    >>> # Rectangular particle
    >>> intensity = parallelepiped_form_factor(qx, qy, qz, 50e-9, 20e-9, 10e-9)
    """
    V = A * B * C
    argx = 0.5 * qx_p * A
    argy = 0.5 * qy_p * B
    argz = 0.5 * qz_p * C
    Fx = jnp.sinc(argx)
    Fy = jnp.sinc(argy)
    Fz = jnp.sinc(argz)
    F = jnp.nan_to_num(delta_rho * Fx * Fy * Fz, nan=1.0)  # *8/V
    return F


@jax.jit
def ellipsoid_formfactor(
    qx: Float[Array, "N_x N_y"],
    qy: Float[Array, "N_x N_y"],
    qz: Float[Array, "N_x N_y"],
    A: float,
    B: float,
    C: float,
    delta_rho: float = 1,
) -> Float[Array, "N_x N_y"]:
    """
    Calculate the analytical form factor for a homogeneous ellipsoid.

    Computes the scattering intensity for an ellipsoid aligned with
    coordinate axes using the analytical solution. The ellipsoid is
    defined by three semi-axes A, B, C along x, y, z directions.

    Parameters
    ----------
    qx, qy, qz : Float[Array, "N_x N_y"]
        Arrays of scattering vector components in particle frame (same shape).
        Should be in units of inverse meters (m⁻¹).
    A, B, C : float
        Semi-axes of the ellipsoid along x, y, z directions in meters.
        The ellipsoid equation is: (x/A)² + (y/B)² + (z/C)² = 1
    delta_rho : float, optional
        Electron density contrast between particle and solvent in
        arbitrary units. Default is 1.0.

    Returns
    -------
    intensity : Float[Array, "N_x N_y"]
        Scattering intensity |F(q)|² with same shape as input q arrays.
        Values represent the scattered intensity at each q-vector.

    Notes
    -----
    - Uses the analytical form: F(q) = V * 3(sin(qR) - qR*cos(qR))/(qR)³
    - Where qR = sqrt((A*qx)² + (B*qy)² + (C*qz)²) is the effective q-radius
    - V = (4/3)π*A*B*C is the ellipsoid volume
    - For a sphere (A=B=C), this reduces to the spherical form factor
    - The function handles the qR=0 case (forward scattering) properly

    Examples
    --------
    >>> # Sphere with 20 nm radius
    >>> intensity = ellipsoid_formfactor(qx, qy, qz, 20e-9, 20e-9, 20e-9)
    >>> # Prolate ellipsoid (elongated along x)
    >>> intensity = ellipsoid_formfactor(qx, qy, qz, 40e-9, 20e-9, 20e-9)
    >>> # Oblate ellipsoid (flattened along z)
    >>> intensity = ellipsoid_formfactor(qx, qy, qz, 30e-9, 30e-9, 10e-9)
    """
    qR = jnp.sqrt((qx * A) ** 2 + (qy * B) ** 2 + (qz * C) ** 2)
    F = jnp.nan_to_num(
        delta_rho * (3 * (jnp.sin(qR) - qR * jnp.cos(qR))) / (qR**3), nan=1.0
    )
    return (F / F.max()) ** 2


@jax.jit
def ellipsoid_formfactor_complex(
    qx: Float[Array, "N_x N_y"],
    qy: Float[Array, "N_x N_y"],
    qz: Float[Array, "N_x N_y"],
    A: float,
    B: float,
    C: float,
    delta_rho: float = 1,
) -> Float[Array, "N_x N_y"]:
    """
    Calculate the analytical form factor for a homogeneous ellipsoid.

    Computes the scattering intensity for an ellipsoid aligned with
    coordinate axes using the analytical solution. The ellipsoid is
    defined by three semi-axes A, B, C along x, y, z directions.

    Parameters
    ----------
    qx, qy, qz : Float[Array, "N_x N_y"]
        Arrays of scattering vector components in particle frame (same shape).
        Should be in units of inverse meters (m⁻¹).
    A, B, C : float
        Semi-axes of the ellipsoid along x, y, z directions in meters.
        The ellipsoid equation is: (x/A)² + (y/B)² + (z/C)² = 1
    delta_rho : float, optional
        Electron density contrast between particle and solvent in
        arbitrary units. Default is 1.0.

    Returns
    -------
    intensity : Float[Array, "N_x N_y"]
        Scattering intensity |F(q)|² with same shape as input q arrays.
        Values represent the scattered intensity at each q-vector.

    Notes
    -----
    - Uses the analytical form: F(q) = V * 3(sin(qR) - qR*cos(qR))/(qR)³
    - Where qR = sqrt((A*qx)² + (B*qy)² + (C*qz)²) is the effective q-radius
    - V = (4/3)π*A*B*C is the ellipsoid volume
    - For a sphere (A=B=C), this reduces to the spherical form factor
    - The function handles the qR=0 case (forward scattering) properly

    Examples
    --------
    >>> # Sphere with 20 nm radius
    >>> intensity = ellipsoid_formfactor(qx, qy, qz, 20e-9, 20e-9, 20e-9)
    >>> # Prolate ellipsoid (elongated along x)
    >>> intensity = ellipsoid_formfactor(qx, qy, qz, 40e-9, 20e-9, 20e-9)
    >>> # Oblate ellipsoid (flattened along z)
    >>> intensity = ellipsoid_formfactor(qx, qy, qz, 30e-9, 30e-9, 10e-9)
    """
    qR = jnp.sqrt((qx * A) ** 2 + (qy * B) ** 2 + (qz * C) ** 2)
    F = jnp.nan_to_num(
        delta_rho * (3 * (jnp.sin(qR) - qR * jnp.cos(qR))) / (qR**3), nan=1.0
    )
    return F


@jax.jit
def rotate_q_space(
    q_xyz: Float[Array, "3 N_x N_y"], R_mat: Float[Array, "3 3"]
) -> Float[Array, "3 N_x N_y"]:
    """
    Rotate the 3D scattering vector space by a given rotation matrix.

    This function applies a 3D rotation transformation to scattering vectors,
    effectively rotating the particle orientation in reciprocal space. This is
    equivalent to rotating the particle in real space before scattering.

    Parameters
    ----------
    q_xyz : Float[Array, "3 N_x N_y"]
        Array of shape (3, N_x, N_y) containing scattering vector components
        in inverse meters. q_xyz[0] = qx, q_xyz[1] = qy, q_xyz[2] = qz.
    R_mat : Float[Array, "3 3"]
        3x3 rotation matrix to apply to the scattering vectors.
        Should be orthogonal with determinant +1.

    Returns
    -------
    q_rotated : Float[Array, "3 N_x N_y"]
        Rotated scattering vectors with same shape as input (3, N_x, N_y).
        The transformation q_rot = R @ q transforms the scattering vectors
        into the rotated particle frame.

    Notes
    -----
    - The function reshapes the 2D detector arrays into 1D for matrix multiplication
    - Rotation is applied as: q_rotated = R_mat @ q_original
    - This is used to simulate different particle orientations in diffraction
    - The rotation matrix typically comes from get_rotation_matrix()

    Examples
    --------
    >>> R = get_rotation_matrix(30, 45, 60)  # Euler angles in degrees
    >>> q_rotated = rotate_q_space(q_vectors, R)
    >>> print(f"Original shape: {q_vectors.shape}, Rotated shape: {q_rotated.shape}")
    """
    q_shape = q_xyz.shape
    q_stack = jnp.stack(
        (q_xyz[0, :, :].ravel(), q_xyz[1, :, :].ravel(), q_xyz[2, :, :].ravel()), axis=0
    )
    q_rot = R_mat @ q_stack
    q = q_rot.reshape(q_shape)
    return q


@jax.jit
def generate_ellipsoid_diffraction(
    q_xyz: Float[Array, "3 N_x N_y"],
    A: float,
    B: float,
    C: float,
    rx: float = 0,
    ry: float = 0,
    rz: float = 0,
) -> Float[Array, "N_x N_y"]:
    """
    Generate diffraction pattern for an ellipsoid with given parameters.

    This function computes the theoretical X-ray scattering intensity for
    a homogeneous ellipsoid with specified dimensions and orientation. The
    calculation combines geometric rotation with analytical form factor computation.

    Parameters
    ----------
    q_xyz : Float[Array, "3 N_x N_y"]
        Array of shape (3, N_x, N_y) containing scattering vector components
        in laboratory frame, in units of inverse meters (m⁻¹).
    A, B, C : float
        Semi-axes of the ellipsoid along x, y, z directions in meters.
        The ellipsoid equation is: (x/A)² + (y/B)² + (z/C)² = 1
    rx, ry, rz : float
        Euler rotation angles in degrees around x, y, z axes respectively.
        Defines the orientation of the ellipsoid relative to the laboratory frame.

    Returns
    -------
    intensity : Float[Array, "N_x N_y"]
        Scattering intensity pattern with shape (N_x, N_y) matching the
        detector geometry. Values represent |F(q)|² where F is the form factor.

    Notes
    -----
    - Uses ZYX Euler angle convention for rotations
    - The intensity is normalized by the maximum value (forward scattering)
    - Handles the q=0 singularity automatically via jnp.nan_to_num
    - Suitable for single-particle diffraction simulation

    Examples
    --------
    >>> # 20x40x40 nm ellipsoid rotated 30° around x-axis
    >>> intensity = generate_ellipsoid_diffraction(q_vectors, 20e-9, 40e-9, 40e-9, 30, 0, 0)
    >>> # Spherical particle (A=B=C)
    >>> intensity = generate_ellipsoid_diffraction(q_vectors, 25e-9, 25e-9, 25e-9, 0, 0, 0)
    """
    q_rot = rotate_q_space(q_xyz, get_rotation_matrix(rx, ry, rz))
    intensity = ellipsoid_formfactor(
        q_rot[0, :, :], q_rot[1, :, :], q_rot[2, :, :], A, B, C
    )
    return intensity


@jax.jit
def generate_ellipsoid_diffraction_complex(
    q_xyz: Float[Array, "3 N_x N_y"],
    A: float,
    B: float,
    C: float,
    rx: float = 0,
    ry: float = 0,
    rz: float = 0,
) -> Complex[Array, "N_x N_y"]:
    """
    Generate diffraction pattern for an ellipsoid with given parameters.

    This function computes the theoretical X-ray scattering intensity for
    a homogeneous ellipsoid with specified dimensions and orientation. The
    calculation combines geometric rotation with analytical form factor computation.

    Parameters
    ----------
    q_xyz : Float[Array, "3 N_x N_y"]
        Array of shape (3, N_x, N_y) containing scattering vector components
        in laboratory frame, in units of inverse meters (m⁻¹).
    A, B, C : float
        Semi-axes of the ellipsoid along x, y, z directions in meters.
        The ellipsoid equation is: (x/A)² + (y/B)² + (z/C)² = 1
    rx, ry, rz : float
        Euler rotation angles in degrees around x, y, z axes respectively.
        Defines the orientation of the ellipsoid relative to the laboratory frame.

    Returns
    -------
    intensity : Float[Array, "N_x N_y"]
        Scattering intensity pattern with shape (N_x, N_y) matching the
        detector geometry. Values represent |F(q)|² where F is the form factor.

    Notes
    -----
    - Uses ZYX Euler angle convention for rotations
    - The intensity is normalized by the maximum value (forward scattering)
    - Handles the q=0 singularity automatically via jnp.nan_to_num
    - Suitable for single-particle diffraction simulation

    Examples
    --------
    >>> # 20x40x40 nm ellipsoid rotated 30° around x-axis
    >>> intensity = generate_ellipsoid_diffraction(q_vectors, 20e-9, 40e-9, 40e-9, 30, 0, 0)
    >>> # Spherical particle (A=B=C)
    >>> intensity = generate_ellipsoid_diffraction(q_vectors, 25e-9, 25e-9, 25e-9, 0, 0, 0)
    """
    q_rot = rotate_q_space(q_xyz, get_rotation_matrix(rx, ry, rz))
    complex_scattering = ellipsoid_formfactor_complex(
        q_rot[0, :, :], q_rot[1, :, :], q_rot[2, :, :], A, B, C
    )
    return complex_scattering


@jax.jit
def generate_sphere_diffraction(q_xyz, r):
    return generate_ellipsoid_diffraction(q_xyz, r, r, r, 0, 0, 0)


@jax.jit
def generate_parallelepiped_diffraction(
    q_xyz: Float[Array, "3 N_x N_y"],
    A: float,
    B: float,
    C: float,
    rx: float = 0,
    ry: float = 0,
    rz: float = 0,
) -> Float[Array, "N_x N_y"]:
    """
    Generate diffraction pattern for a parallelepiped with given parameters.

    This function computes the theoretical X-ray scattering intensity for
    a homogeneous rectangular parallelepiped (box) with specified dimensions
    and orientation. The calculation uses analytical form factors for efficiency.

    Parameters
    ----------
    q_xyz : Float[Array, "3 N_x N_y"]
        Array of shape (3, N_x, N_y) containing scattering vector components
        in laboratory frame, in units of inverse meters (m⁻¹).
    A, B, C : float
        Side lengths of the parallelepiped along x, y, z axes in meters.
        A is x-dimension, B is y-dimension, C is z-dimension.
    rx, ry, rz : float
        Euler rotation angles in degrees around x, y, z axes respectively.
        Defines the orientation of the box relative to the laboratory frame.

    Returns
    -------
    intensity : Float[Array, "N_x N_y"]
        Scattering intensity pattern with shape (N_x, N_y) matching the
        detector geometry. Values represent |F(q)|² where F is the form factor.

    Notes
    -----
    - Uses ZYX Euler angle convention for rotations
    - Form factor based on product of sinc functions along each axis
    - Shows characteristic oscillations and sharp minima due to box edges
    - The intensity is normalized by the maximum value (forward scattering)
    - Suitable for crystalline or rectangular nanoparticle simulation

    Examples
    --------
    >>> # 20x20x20 nm cube
    >>> intensity = generate_parallelepiped_diffraction(q_vectors, 20e-9, 20e-9, 20e-9, 0, 0, 0)
    >>> # Rectangular rod rotated 45° around z-axis
    >>> intensity = generate_parallelepiped_diffraction(q_vectors, 50e-9, 10e-9, 10e-9, 0, 0, 45)
    """
    q_rot = rotate_q_space(q_xyz, get_rotation_matrix(rx, ry, rz))
    intensity = parallelepiped_form_factor(
        q_rot[0, :, :], q_rot[1, :, :], q_rot[2, :, :], A, B, C
    )
    return intensity


@jax.jit
def generate_parallelepiped_diffraction_complex(
    q_xyz: Float[Array, "3 N_x N_y"],
    A: float,
    B: float,
    C: float,
    rx: float = 0,
    ry: float = 0,
    rz: float = 0,
) -> Complex[Array, "N_x N_y"]:
    """
    Generate diffraction pattern for a parallelepiped with given parameters.

    This function computes the theoretical X-ray scattering intensity for
    a homogeneous rectangular parallelepiped (box) with specified dimensions
    and orientation. The calculation uses analytical form factors for efficiency.

    Parameters
    ----------
    q_xyz : Float[Array, "3 N_x N_y"]
        Array of shape (3, N_x, N_y) containing scattering vector components
        in laboratory frame, in units of inverse meters (m⁻¹).
    A, B, C : float
        Side lengths of the parallelepiped along x, y, z axes in meters.
        A is x-dimension, B is y-dimension, C is z-dimension.
    rx, ry, rz : float
        Euler rotation angles in degrees around x, y, z axes respectively.
        Defines the orientation of the box relative to the laboratory frame.

    Returns
    -------
    intensity : Float[Array, "N_x N_y"]
        Scattering intensity pattern with shape (N_x, N_y) matching the
        detector geometry. Values represent |F(q)|² where F is the form factor.

    Notes
    -----
    - Uses ZYX Euler angle convention for rotations
    - Form factor based on product of sinc functions along each axis
    - Shows characteristic oscillations and sharp minima due to box edges
    - The intensity is normalized by the maximum value (forward scattering)
    - Suitable for crystalline or rectangular nanoparticle simulation

    Examples
    --------
    >>> # 20x20x20 nm cube
    >>> intensity = generate_parallelepiped_diffraction(q_vectors, 20e-9, 20e-9, 20e-9, 0, 0, 0)
    >>> # Rectangular rod rotated 45° around z-axis
    >>> intensity = generate_parallelepiped_diffraction(q_vectors, 50e-9, 10e-9, 10e-9, 0, 0, 45)
    """
    q_rot = rotate_q_space(q_xyz, get_rotation_matrix(rx, ry, rz))
    complex_scattering = parallelepiped_form_factor_complex(
        q_rot[0, :, :], q_rot[1, :, :], q_rot[2, :, :], A, B, C
    )
    return complex_scattering


def fib_lattice_sq(
    n_points: int, phi: float = (1 + jnp.sqrt(5)) / 2, epsilon: float = 0.5
) -> Float[Array, "2 n_points"]:
    """
    Generate points on a square Fibonacci lattice.

    This function creates a quasi-uniform distribution of points in the unit square
    [0,1]×[0,1] using the Fibonacci sequence. The resulting pattern has low
    discrepancy and is useful for quasi-Monte Carlo sampling or uniform coverage
    of 2D parameter spaces.

    Parameters
    ----------
    n_points : int
        Number of points to generate in the lattice
    phi : float, optional
        Golden ratio (1 + √5)/2 ≈ 1.618. Used as the irrational number for
        the quasi-random sequence generation. Default is the golden ratio.
    epsilon : float, optional
        Small offset parameter to avoid boundary effects. Controls the
        distribution near the edges of the unit square. Default is 0.5.

    Returns
    -------
    points : Float[Array, "2 n_points"]
        Array of shape (2, n_points) containing 2D coordinates in [0,1]×[0,1].
        points[0, :] are x-coordinates, points[1, :] are y-coordinates.

    Notes
    -----
    - Uses the golden ratio to ensure low discrepancy (quasi-uniform distribution)
    - The sequence is deterministic and reproducible for the same n_points
    - Forms the basis for higher-dimensional Fibonacci lattices
    - Epsilon parameter helps avoid clustering near boundaries

    Examples
    --------
    >>> points = fib_lattice_sq(100)
    >>> print(f"Generated {points.shape[1]} points in unit square")
    >>> # Use for parameter sampling
    >>> x_params = points[0, :] * (x_max - x_min) + x_min
    >>> y_params = points[1, :] * (y_max - y_min) + y_min
    """
    indices = jnp.arange(0, n_points, dtype=jnp.float32)

    x = jnp.mod(indices / phi, 1.0)
    y = (indices + epsilon) / (n_points - 1 + 2 * epsilon)

    return jnp.stack((x, y))


def fib_lattice_disc(
    n_points: int, phi: float = (1 + jnp.sqrt(5)) / 2
) -> Float[Array, "2 n_points"]:
    """
    Generate points on a disc using Fibonacci lattice in polar coordinates.

    This function creates a quasi-uniform distribution of points within a unit disc
    by transforming a square Fibonacci lattice to polar coordinates. The radial
    coordinate is square-root transformed to ensure uniform area density.

    Parameters
    ----------
    n_points : int
        Number of points to generate on the disc
    phi : float, optional
        Golden ratio (1 + √5)/2 ≈ 1.618. Used for the underlying square lattice
        generation. Default is the golden ratio.

    Returns
    -------
    polar_points : Float[Array, "2 n_points"]
        Array of shape (2, n_points) containing polar coordinates.
        polar_points[0, :] are radial coordinates r ∈ [0, 1]
        polar_points[1, :] are angular coordinates θ ∈ [0, 2π]

    Notes
    -----
    - Radial coordinates use √r transformation for uniform area distribution
    - Angular coordinates span the full circle [0, 2π]
    - Maintains the low-discrepancy properties of Fibonacci lattices
    - Useful for sampling orientations or positions within circular regions

    Examples
    --------
    >>> polar_coords = fib_lattice_disc(500)
    >>> r, theta = polar_coords[0, :], polar_coords[1, :]
    >>> # Convert to Cartesian if needed
    >>> x = r * jnp.cos(theta)
    >>> y = r * jnp.sin(theta)
    """
    (
        theta,
        ro,
    ) = fib_lattice_sq(n_points, phi)
    ro, theta = jnp.sqrt(ro), theta * 2 * jnp.pi

    return jnp.stack((ro, theta))


def fib_lattice_sphere_theta_phi(
    n_points: int, phi: float = (1 + jnp.sqrt(5)) / 2
) -> Float[Array, "2 n_points"]:
    """
    Generate points on a unit sphere using spherical Fibonacci lattice.

    This function creates a quasi-uniform distribution of points on the surface
    of a unit sphere by transforming a square Fibonacci lattice to spherical
    coordinates. The resulting distribution has excellent uniformity properties
    for spherical sampling applications.

    Parameters
    ----------
    n_points : int
        Number of points to generate on the sphere surface
    phi : float, optional
        Golden ratio (1 + √5)/2 ≈ 1.618. Used for the underlying square lattice
        generation. Default is the golden ratio.

    Returns
    -------
    sphere_points : Float[Array, "2 n_points"]
        Array of shape (2, n_points) containing spherical coordinates
        on the unit sphere. All points satisfy θ ∈ [0, 2π] and φ ∈ [0, π].
        sphere_points[0, :] are theta-coordinates
        sphere_points[1, :] are phi-coordinates


    Notes
    -----
    - Uses equal-area projection to maintain uniform distribution on sphere
    - Azimuthal angle θ ∈ [0, 2π] and polar angle φ ∈ [0, π]
    - The cos(φ) transformation ensures uniform area density
    - Excellent for sampling particle orientations in 3D scattering

    Examples
    --------
    >>> sphere_coords = fib_lattice_sphere(1000)
    >>> x, y, z = sphere_coords[0, :], sphere_coords[1, :], sphere_coords[2, :]
    >>> # Verify points are on unit sphere
    >>> assert jnp.allclose(x**2 + y**2 + z**2, 1.0)
    >>> # Use for orientation sampling
    >>> orientations = sphere_coords * max_rotation_angle
    """
    (
        x,
        y,
    ) = fib_lattice_sq(n_points, phi)
    theta = x * 2 * jnp.pi
    phi_ = jnp.arccos(1 - 2 * y)

    return jnp.stack(
        (
            theta,
            phi_,
        )
    )


def fib_lattice_sphere_xyz(
    n_points: int, phi: float = (1 + jnp.sqrt(5)) / 2
) -> Float[Array, "3 n_points"]:
    """
    Generate points on a unit sphere using spherical Fibonacci lattice.

    This function creates a quasi-uniform distribution of points on the surface
    of a unit sphere by transforming a square Fibonacci lattice to spherical
    coordinates. The resulting distribution has excellent uniformity properties
    for spherical sampling applications.

    Parameters
    ----------
    n_points : int
        Number of points to generate on the sphere surface
    phi : float, optional
        Golden ratio (1 + √5)/2 ≈ 1.618. Used for the underlying square lattice
        generation. Default is the golden ratio.

    Returns
    -------
    sphere_points : Float[Array, "3 n_points"]
        Array of shape (3, n_points) containing 3D Cartesian coordinates
        on the unit sphere. All points satisfy x² + y² + z² = 1.
        sphere_points[0, :] are x-coordinates
        sphere_points[1, :] are y-coordinates
        sphere_points[2, :] are z-coordinates

    Notes
    -----
    - Uses equal-area projection to maintain uniform distribution on sphere
    - Azimuthal angle θ ∈ [0, 2π] and polar angle φ ∈ [0, π]
    - The cos(φ) transformation ensures uniform area density
    - Excellent for sampling particle orientations in 3D scattering

    Examples
    --------
    >>> sphere_coords = fib_lattice_sphere(1000)
    >>> x, y, z = sphere_coords[0, :], sphere_coords[1, :], sphere_coords[2, :]
    >>> # Verify points are on unit sphere
    >>> assert jnp.allclose(x**2 + y**2 + z**2, 1.0)
    >>> # Use for orientation sampling
    >>> orientations = sphere_coords * max_rotation_angle
    """
    (
        x,
        y,
    ) = fib_lattice_sq(n_points, phi)
    theta = x * 2 * jnp.pi
    phi_ = jnp.arccos(1 - 2 * y)

    x = jnp.cos(theta) * jnp.sin(phi_)
    y = jnp.sin(theta) * jnp.sin(phi_)
    z = jnp.cos(phi_)

    return jnp.stack((x, y, z))


def generate_spaced_rotations(points_per_sphere: int = 100, in_plane_steps: int = 30):
    rotation_theta_phi = fib_lattice_sphere_theta_phi(points_per_sphere)

    symmetry_mask = (rotation_theta_phi[0, :] < jnp.pi / 2) * (
        rotation_theta_phi[1, :] < jnp.pi / 2
    )

    rotation_theta_phi = rotation_theta_phi[:, symmetry_mask]

    rotation_omega: Array = jnp.radians(
        jnp.linspace(0, 180, in_plane_steps, endpoint=False)
    )
    # print(rotation_theta_phi.shape, rotation_omega.shape)
    # print(jnp.repeat(rotation_theta_phi, rotation_omega.shape[0],axis=1).shape,
    # jnp.tile(rotation_omega, rotation_theta_phi.shape[1]).shape)
    return jnp.vstack(
        (
            jnp.repeat(rotation_theta_phi, rotation_omega.shape[0], axis=1),
            jnp.tile(rotation_omega, rotation_theta_phi.shape[1]),
        )
    )


def chunked(lst, size):
    """Yield successive `size`-element chunks from list `lst`."""
    for i in range(0, len(lst), size):
        yield lst[i : i + size]
