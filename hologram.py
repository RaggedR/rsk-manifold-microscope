"""
The Mathematics of Holography — Phase, Interference, and the Fourier Transform

This builds up the math of holography from first principles, one concept at a time.

Key idea: light propagation through free space IS a Fourier transform.
A hologram works because it records phase (via interference), and
reconstruction is the inverse Fourier transform.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.widgets import Slider


def complex_to_hsv(z):
    """
    Visualise a complex field as colour:
      - Hue = phase (angle in the complex plane)
      - Brightness = amplitude (magnitude)

    This is the key visualisation: it shows BOTH amplitude and phase
    in a single image. A photograph only captures brightness (amplitude).
    A hologram captures both.
    """
    amp = np.abs(z)
    phase = np.angle(z)  # -pi to pi

    H = (phase + np.pi) / (2 * np.pi)  # map to [0, 1]
    S = np.ones_like(H)
    V = amp / (amp.max() + 1e-10)      # normalise brightness
    V = np.clip(V, 0, 1)

    hsv = np.stack([H, S, V], axis=-1)
    return hsv_to_rgb(hsv)


def run():
    N = 512
    L = 1.0  # dimensionless units for clarity
    dx = L / N
    x = np.linspace(-L/2, L/2, N, endpoint=False)
    xx, yy = np.meshgrid(x, x)

    # Spatial frequency coordinates (Fourier domain)
    fx = np.fft.fftfreq(N, d=dx)
    FX, FY = np.fft.fftshift(np.meshgrid(fx, fx))

    # === The object: a simple shape ===
    # A cross shape — easy to recognise in both spatial and Fourier domains
    obj = np.zeros((N, N), dtype=complex)
    cross_h = (np.abs(yy) < 0.03) & (np.abs(xx) < 0.15)
    cross_v = (np.abs(xx) < 0.03) & (np.abs(yy) < 0.15)
    obj[cross_h | cross_v] = 1.0

    # Initial propagation distance
    z0 = 50.0  # propagation distance (in units of dx)

    # === Compute all stages ===
    def compute_stages(z):
        """
        Compute the full holographic pipeline at propagation distance z.
        Returns dict of all intermediate fields.
        """
        # STAGE 1: Object field (what we want to record)
        # This is just the object — a real-valued amplitude, phase = 0 everywhere

        # STAGE 2: Propagated field (object field after traveling distance z)
        #
        # THIS IS WHERE THE FOURIER TRANSFORM ENTERS.
        #
        # The angular spectrum method says:
        #   1. Take the 2D FT of the field:  F = FT[E(x,y)]
        #   2. Multiply by the free-space transfer function:
        #        H(fx,fy) = exp(i * 2pi * z * sqrt(1/λ² - fx² - fy²))
        #   3. Inverse FT:  E'(x,y) = IFT[F · H]
        #
        # In our dimensionless units (λ=1), this simplifies to:
        #        H(fx,fy) = exp(i * 2pi * z * sqrt(1 - fx² - fy²))
        #
        # The transfer function H is a PURE PHASE — it has |H| = 1 everywhere.
        # It doesn't change amplitudes, it only rotates phases.
        # Different spatial frequencies (angles) get different phase shifts.
        # That's literally all that "propagation" is: differential phase accumulation.

        obj_spectrum = np.fft.fft2(obj)  # Step 1: FT

        sq = 1.0 - (fx[:, None]**2 + fx[None, :]**2)  # 1 - fx² - fy²  (using fftfreq coords)
        propagating = sq >= 0
        H = np.zeros((N, N), dtype=complex)
        H[propagating] = np.exp(1j * 2 * np.pi * z * np.sqrt(sq[propagating]))

        prop_spectrum = obj_spectrum * H  # Step 2: multiply
        prop_field = np.fft.ifft2(prop_spectrum)  # Step 3: IFT

        # STAGE 3: Reference beam — a tilted plane wave
        # Phase increases linearly in x (that's what "tilted" means)
        ref_angle = 0.05  # small tilt
        ref = np.exp(1j * 2 * np.pi * ref_angle * xx / dx * 0.1)

        # STAGE 4: Hologram = |E_prop + E_ref|²
        # This is the KEY TRICK. We can't record complex fields (phase).
        # But we CAN record intensity (amplitude²).
        # Expanding: |O + R|² = |R|² + |O|² + R*O + RO*
        #                        ^^^    ^^^   ^^^   ^^^
        #                        DC     DC   signal conjugate
        # The R*O term contains the FULL complex field O, multiplied by R*.
        # Recording intensity converts phase into amplitude variations.
        hologram = np.abs(prop_field + ref) ** 2

        # What a photograph would record (no reference beam — just |O|²):
        photograph = np.abs(prop_field) ** 2

        # STAGE 5: Reconstruction
        # Illuminate hologram with reference beam, then propagate BACKWARDS (-z)
        illuminated = hologram * ref

        illum_spectrum = np.fft.fft2(illuminated)
        H_back = np.zeros((N, N), dtype=complex)
        H_back[propagating] = np.exp(1j * 2 * np.pi * (-z) * np.sqrt(sq[propagating]))
        recon_field = np.fft.ifft2(illum_spectrum * H_back)

        # Also try reconstructing from the photograph (should fail)
        photo_illum = photograph * ref
        photo_spectrum = np.fft.fft2(photo_illum)
        photo_recon = np.fft.ifft2(photo_spectrum * H_back)

        return {
            'object': obj,
            'obj_spectrum': np.fft.fftshift(obj_spectrum),
            'transfer_fn': np.fft.fftshift(H),
            'prop_field': prop_field,
            'prop_spectrum': np.fft.fftshift(prop_spectrum),
            'reference': ref,
            'hologram': hologram,
            'photograph': photograph,
            'recon_field': recon_field,
            'photo_recon': photo_recon,
        }

    stages = compute_stages(z0)

    # === FIGURE: The mathematical pipeline ===
    fig = plt.figure(figsize=(16, 11), facecolor='#0d1117')
    fig.suptitle("The Mathematics of Holography",
                 fontsize=16, fontweight='bold', color='white', y=0.98)

    # Layout: 3 rows x 4 columns
    # Row 1: Object → FT → Transfer Function → Propagated Field
    # Row 2: Reference beam → Hologram (|O+R|²) → vs Photograph (|O|²)
    # Row 3: Reconstruction from hologram vs reconstruction from photograph

    ax_specs = [
        # Row 1: Propagation = Fourier transform
        (341, "① Object\n(amplitude only, no phase)"),
        (342, "② Fourier Transform\n(spatial frequency spectrum)"),
        (343, "③ Transfer Function H(fx,fy)\n(pure phase — rotates, doesn't scale)"),
        (344, "④ Propagated Field\n(now has complex phase!)"),
        # Row 2: Recording
        (345, "⑤ Reference Beam\n(known phase pattern)"),
        (346, "⑥ Hologram: |Object + Ref|²\n(interference encodes phase as intensity)"),
        (347, "⑦ Photograph: |Object|²\n(intensity only — phase lost!)"),
        (348, ""),
        # Row 3: Reconstruction
        (3, 4, 9, "⑧ Reconstructed from HOLOGRAM\n(cross recovers — phase was preserved)"),
        (3, 4, 10, "⑨ Reconstructed from PHOTOGRAPH\n(no recovery — phase was lost)"),
        (3, 4, 11, "⑩ Phase difference\n(hologram kept this; photo didn't)"),
        (3, 4, 12, ""),
    ]

    extent = [-L/2, L/2, -L/2, L/2]
    fextent = [-N/(2*L), N/(2*L), -N/(2*L), N/(2*L)]

    axes = []

    def style_ax(ax, title):
        ax.set_title(title, fontsize=8, color='white', pad=6)
        ax.tick_params(colors='white', labelsize=6)
        for sp in ax.spines.values():
            sp.set_edgecolor('#333')

    # --- Row 1: The Fourier Transform story ---
    ax1 = fig.add_subplot(3, 4, 1)
    style_ax(ax1, "① Object\n(real-valued, no phase)")
    im1 = ax1.imshow(np.abs(stages['object']), cmap='inferno', extent=extent)
    ax1.set_xlabel("x", color='grey', fontsize=7)

    ax2 = fig.add_subplot(3, 4, 2)
    style_ax(ax2, "② FT of Object\n(Fourier spectrum)")
    spec = np.log1p(np.abs(stages['obj_spectrum']))
    im2 = ax2.imshow(spec, cmap='inferno', extent=fextent)
    ax2.set_xlabel("fx (spatial freq)", color='grey', fontsize=7)

    ax3 = fig.add_subplot(3, 4, 3)
    style_ax(ax3, "③ Transfer Function H\n(phase only — this IS propagation)")
    im3 = ax3.imshow(complex_to_hsv(stages['transfer_fn']), extent=fextent)
    ax3.set_xlabel("fx", color='grey', fontsize=7)

    ax4 = fig.add_subplot(3, 4, 4)
    style_ax(ax4, "④ Propagated Field\n(colour = phase, brightness = amplitude)")
    im4 = ax4.imshow(complex_to_hsv(stages['prop_field']), extent=extent)
    ax4.set_xlabel("x", color='grey', fontsize=7)

    # --- Row 2: Recording —  hologram vs photograph ---
    ax5 = fig.add_subplot(3, 4, 5)
    style_ax(ax5, "⑤ Reference Beam\n(tilted plane wave — known phase)")
    im5 = ax5.imshow(complex_to_hsv(stages['reference']), extent=extent)

    ax6 = fig.add_subplot(3, 4, 6)
    style_ax(ax6, "⑥ HOLOGRAM: |Obj + Ref|²\n(interference captures phase)")
    im6 = ax6.imshow(stages['hologram'], cmap='gray', extent=extent)

    ax7 = fig.add_subplot(3, 4, 7)
    style_ax(ax7, "⑦ PHOTOGRAPH: |Obj|²\n(just intensity — phase is gone)")
    im7 = ax7.imshow(stages['photograph'], cmap='gray', extent=extent)

    # Phase information panel
    ax8 = fig.add_subplot(3, 4, 8)
    style_ax(ax8, "Phase of propagated field\n(this is what the hologram saves)")
    im8 = ax8.imshow(np.angle(stages['prop_field']), cmap='twilight', extent=extent)
    ax8.set_xlabel("x", color='grey', fontsize=7)

    # --- Row 3: Reconstruction ---
    ax9 = fig.add_subplot(3, 4, 9)
    style_ax(ax9, "⑧ Reconstruction from HOLOGRAM\n(the cross comes back!)")
    r_holo = np.abs(stages['recon_field'])
    im9 = ax9.imshow(r_holo, cmap='hot', extent=extent)
    im9.set_clim(0, np.percentile(r_holo, 99.5))

    ax10 = fig.add_subplot(3, 4, 10)
    style_ax(ax10, "⑨ Reconstruction from PHOTOGRAPH\n(no recovery — just blur)")
    r_photo = np.abs(stages['photo_recon'])
    im10 = ax10.imshow(r_photo, cmap='hot', extent=extent)
    im10.set_clim(0, np.percentile(r_photo, 99.5))

    ax11 = fig.add_subplot(3, 4, 11)
    style_ax(ax11, "⑩ Hologram vs Photo difference\n(what phase preservation buys you)")
    diff = r_holo / (r_holo.max() + 1e-10) - r_photo / (r_photo.max() + 1e-10)
    im11 = ax11.imshow(diff, cmap='RdBu', extent=extent, vmin=-0.5, vmax=0.5)

    # Equation summary panel
    ax12 = fig.add_subplot(3, 4, 12)
    ax12.set_facecolor('#0d1117')
    ax12.axis('off')
    eqs = (
        "The key equations:\n\n"
        "Propagation:\n"
        "  E'(x) = FT⁻¹[ FT[E(x)] · H(f) ]\n\n"
        "Transfer function:\n"
        "  H(f) = exp(i·2π·z·√(1-fx²-fy²))\n"
        "  |H| = 1  (pure phase!)\n\n"
        "Hologram records:\n"
        "  I = |E + R|²\n"
        "    = |R|² + |E|² + R*E + RE*\n"
        "                    ↑\n"
        "          This term has the\n"
        "          full complex field E\n\n"
        "Photo records:\n"
        "  I = |E|²  (phase lost forever)"
    )
    ax12.text(0.05, 0.95, eqs, transform=ax12.transAxes,
              fontsize=7.5, color='#cccccc', fontfamily='monospace',
              verticalalignment='top')

    plt.subplots_adjust(bottom=0.10, top=0.91, hspace=0.5, wspace=0.3,
                        left=0.04, right=0.97)

    # --- Propagation distance slider ---
    ax_slider = fig.add_axes([0.15, 0.03, 0.70, 0.02], facecolor='#1a1a2e')
    slider = Slider(ax_slider, 'Propagation z',
                    1, 150, valinit=z0, color='#e94560')
    slider.label.set_color('white')
    slider.valtext.set_color('white')

    def update(val):
        z = val
        s = compute_stages(z)

        im3.set_data(complex_to_hsv(s['transfer_fn']))
        im4.set_data(complex_to_hsv(s['prop_field']))
        im5.set_data(complex_to_hsv(s['reference']))
        im6.set_data(s['hologram'])
        im6.set_clim(s['hologram'].min(), s['hologram'].max())
        im7.set_data(s['photograph'])
        im7.set_clim(s['photograph'].min(), s['photograph'].max())
        im8.set_data(np.angle(s['prop_field']))

        rh = np.abs(s['recon_field'])
        im9.set_data(rh)
        im9.set_clim(0, np.percentile(rh, 99.5))

        rp = np.abs(s['photo_recon'])
        im10.set_data(rp)
        im10.set_clim(0, np.percentile(rp, 99.5))

        d = rh / (rh.max() + 1e-10) - rp / (rp.max() + 1e-10)
        im11.set_data(d)

        fig.canvas.draw_idle()

    slider.on_changed(update)

    # Row labels
    fig.text(0.01, 0.83, "PROPAGATION\n(= Fourier\ntransform)",
             fontsize=8, color='#e94560', fontweight='bold',
             rotation=0, va='center')
    fig.text(0.01, 0.50, "RECORDING\n(hologram\nvs photo)",
             fontsize=8, color='#e94560', fontweight='bold',
             rotation=0, va='center')
    fig.text(0.01, 0.20, "RECONSTRUCTION\n(inverse FT\nback to object)",
             fontsize=8, color='#e94560', fontweight='bold',
             rotation=0, va='center')

    plt.show()


if __name__ == "__main__":
    run()
