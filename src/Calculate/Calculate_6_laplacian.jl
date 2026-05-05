export laplacian!

function laplacian!(
    lap::ScalarField,
    phi::ScalarField,
    config;
    scale_factor = one(eltype(phi.values))
)
    (; hardware) = config
    (; backend, workgroup) = hardware

    # Interior kernel: one thread per cell.
    #
    # Boundary faces are not detected by face ID. Instead, inside the kernel we
    # only use faces that have a valid neighbouring cell across the face.
    ndrange = length(lap)
    ncells  = length(phi.values)

    kernel! = _laplacian_interior!(_setup(backend, workgroup, ndrange)...)
    kernel!(lap, phi, scale_factor, ncells)

    return lap
end


# Laplacian kernel definition
@kernel function _laplacian_interior!(lap, phi, scale_factor, ncells)
    cID = @index(Global)

    @uniform begin
        (; mesh) = phi
        (; faces, cells, cell_faces, cell_nsign) = mesh

        z = zero(eltype(phi.values))
    end

    @inbounds begin
        (; volume, faces_range) = cells[cID]

        phiP = phi.values[cID]
        res  = z

        for fi ∈ faces_range
            fID = cell_faces[fi]

            face = faces[fID]
            ownerCells = face.ownerCells

            # Boundary faces may have only one real owner, or may store an
            # invalid/dummy second owner. Either way, they must not enter the
            # internal cell-pair stencil.
            length(ownerCells) < 2 && continue

            c1 = ownerCells[1]
            c2 = ownerCells[2]

            # Identify the neighbouring cell across this face.
            cN = ifelse(cID == c1, c2, c1)

            # If the neighbour is not a real cell, this is a boundary face.
            cN < 1 && continue
            cN > ncells && continue
            cN == cID && continue

            phiN = phi.values[cN]

            nsign = cell_nsign[fi]
            (; area, normal, delta, e) = face

            # Oriented face-area vector for this cell.
            Sf = nsign * area * normal

            # Oriented cell-to-cell unit vector for this cell.
            ef = nsign * e

            # Minimum-correction orthogonal contribution.
            Ef     = ((Sf ⋅ Sf) / (Sf ⋅ ef)) * ef
            Ef_mag = norm(Ef)

            res += scale_factor * (Ef_mag / delta) * (phiN - phiP)
        end

        lap.values[cID] = res / volume
    end
end