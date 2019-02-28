import tensorflow as tf


def symmetric_matrix_from_eig_decomp(eig_vals, eig_vec, name='symmetric_matrix_from_eig_decomp', do_inv=False,
                                     out_name=None):
    # Build a matrix m = eig_vec * diag(eig_vals) * eig_vec^T
    # If do_inv compute m^-1 instead, eig_vals should be larger than 0
    with tf.name_scope(name):
        if do_inv:
            eig_vals = 1.0 / eig_vals
        eig_vals_diag = tf.matrix_diag(eig_vals)
        matrix = tf.matmul(tf.matmul(eig_vec, eig_vals_diag), eig_vec, transpose_b=True, name=out_name)

    return matrix


def symmetric_matrix_from_eig_decomp_with_diag(eig_vals, eig_vec, diag_add,
                                               name='symmetric_matrix_from_eig_decomp_with_diag', do_inv=False,
                                               out_name=None):
    # Build a matrix m = eig_vec * diag(eig_vals) * eig_vec^T + diag_add
    # If do_inv compute m^-1 instead, eig_vals and diag_add should be larger than 0
    with tf.name_scope(name):
        if not do_inv:
            eig_vals_diag = tf.matrix_diag(eig_vals)
            matrix = tf.matmul(tf.matmul(eig_vec, eig_vals_diag), eig_vec, transpose_b=True)
            matrix = tf.add(matrix, tf.matrix_diag(diag_add), name=out_name)
        else:
            # Use the Woodbury identity to do only the inverse of a matrix as big as the number of eigen vectors
            # (a + cbc^t)^-1 = a^-1 - a^-1 c (b^-1 + c^t a^1 c)^-1 c^t a^-1
            a = tf.matrix_diag(1.0 / diag_add)
            b = tf.matrix_diag(1.0 / eig_vals)
            c = eig_vec

            ac = tf.matmul(a, c)
            b_cac = tf.matrix_inverse(b + tf.matmul(c, ac, transpose_a=True))
            ac_b_cac_c = tf.matmul(tf.matmul(ac, b_cac), c, transpose_b=True)
            matrix = tf.subtract(a, tf.matmul(ac_b_cac_c, a), name=out_name)

    return matrix


def _make_matrix_orthonormal_gram_schmidt(m):
    # Orthonormalize a matrix with the stabilized Gram-Schmidt process
    # Arfken, G. "Gram-Schmidt Orthogonalization." 1985
    # Iteratively project each vector to be orthogonal to the previous ones
    # Cost is O(nk^2), where n is number of rows and k is number of columns (vectors to orthonormalize)
    # This code was adapted from a Matlab implementation, transpose to work with columns and undo at the end
    m = tf.transpose(m, perm=[0, 2, 1])
    k = m.shape[1]
    m_ortho = [None] * k

    for i in range(0, k):
        m_ortho[i] = tf.expand_dims(m[:, i], axis=1)
        for j in range(0, i):
            m_ortho_ij = tf.matmul(m_ortho[i], m_ortho[j], transpose_b=True)
            m_ortho_jj = tf.matmul(m_ortho[j], m_ortho[j], transpose_b=True)
            m_ortho[i] -= (m_ortho_ij / m_ortho_jj) * m_ortho[j]
        m_ortho[i] /= tf.norm(m_ortho[i], axis=2, keep_dims=True)

    m_ortho = tf.concat(m_ortho, axis=1)
    m_ortho = tf.transpose(m_ortho, perm=[0, 2, 1])
    return m_ortho


def _make_matrix_orthonormal_householder(_):
    # Orthonormalize a matrix with the Householder method
    raise NotImplementedError()


def make_matrix_orthonormal(m, method='gram_schmidt', name='make_matrix_orthonormal'):
    # Orthonormalize the input matrix m, such that m_output.T * m_output = I
    # Each column will contain an orthonormal vector
    with tf.name_scope(name):

        added_batch_dim = False

        if len(m.shape) == 2:
            # Add batch dimension
            m = tf.expand_dims(m, 0)
            added_batch_dim = True

        assert len(m.shape) == 3, "Input must be rank 2 or 3"
        assert m.shape[1] is not None, "Row shape must be defined"
        assert m.shape[2] is not None, "Column shape must be defined"

        if method == 'gram_schmidt':
            m_ortho = _make_matrix_orthonormal_gram_schmidt(m)
        elif method == 'householder':
            m_ortho = _make_matrix_orthonormal_householder(m)
        else:
            raise RuntimeError("Unkown method for orthogonalization")

        if added_batch_dim:
            # Remove batch dimension
            m_ortho = tf.squeeze(m_ortho, axis=0)

        return m_ortho


def sqrtm_eig(eig_vals, eig_vec, name='sqrtm_eig'):
    # Matrix square root, returns matrix A such that A*A=M, where M is the input matrix given its eigen decomposition
    with tf.name_scope(name=name):
        sqrt_eig_vals = tf.sqrt(eig_vals)
        return symmetric_matrix_from_eig_decomp(eig_vals=sqrt_eig_vals, eig_vec=eig_vec)


def sqrtm_h(m, name='sqrtm_h'):
    # Matrix square root of a Hermitian matrix, returns matrix A such that A*A=M, where M is the input matrix
    with tf.name_scope(name=name):
        eig_vals, eig_vec = tf.self_adjoint_eig(m)
        return sqrtm_eig(eig_vals=eig_vals, eig_vec=eig_vec)
