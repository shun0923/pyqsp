import qulacs
import numpy as np


class QSPCircuit(qulacs.QuantumCircuit):
    """QSP circuit

    A `qulacs.QuantumCircuit` that implements the QSP sequence given by `phis`

    A tool to evaluate and visualize the response of a given QSP sequence.

    Allows substitution of arbitrary theta into the sequence
    """

    def __init__(self, phis):
        super(QSPCircuit, self).__init__(1)
        # recall that in the QSP sequence we rotate as exp(i * phi * Z), but
        # rz(theta) := exp(i * theta/2 * Z)
        self.phis = np.array(phis).flatten() * 2
        self.q = 0

    def build_qsp_sequence(self, theta):
        self.add_RZ_gate(self.q, self.phis[0])
        for phi in self.phis[1:]:
            self.add_RX_gate(self.q, theta)
            self.add_RZ_gate(self.q, phi)

    def svg(self):
        """Get the SVG circuit (for visualization)"""
        return None  # SVGCircuit(self)

    def qsp_response(self, thetas):
        """Evaluate the QSP response for a list of thetas

        params
        -----
        thetas: list of floats
            list of theta input of a QSP sequence

        returns
        -------
        numpy array with shape (len(params),)
            evaluates the qsp response Re[P(x)] + i * Re[Q(x)] * sqrt(1-x^2) from post selecting on |+> for each theta in thetas
        """
        return np.real(self.eval_px(thetas)) + \
            1j * np.real(self.eval_qx(thetas)) * np.sin(thetas)

    def eval_px(self, thetas):
        """Evaluate P(x) for a list of thetas

        params
        -----
        thetas: list of floats
            list of theta input of a QSP sequence

        returns
        -------
        numpy array with shape (len(params),)
            evaluates P(x) from the resulting QSP sequence for each theta in thetas
        """
        pxs = []
        for theta in np.array(thetas).flatten():
            circ = self.build_qsp_sequence(theta * 2)
            state = qulacs.QuantumState(1)
            state.set_zero_state()
            circ.update_quantum_state(state)
            pxs.append(state.get_amplitude(0))
        return np.array(pxs)

    def eval_real_px(self, thetas):
        """Evaluate the QSP response (real part) for a list of thetas"""
        return np.real(self.eval_px(thetas))

    def eval_imag_px(self, thetas):
        """Evaluate the QSP response (imaginary part) for a list of thetas"""
        return np.imag(self.eval_px(thetas))

    def eval_qx(self, thetas):
        """Evaluate Q(x) for a list of thetas

        params
        -----
        thetas: list of floats
            list of theta input of a QSP sequence

        returns
        -------
        numpy array with shape (len(params),)
            evaluates Q(x) from the resulting QSP sequence for each theta in thetas
        """
        qxs = []
        for theta in np.array(thetas).flatten():
            circ = self.build_qsp_sequence(theta * 2)
            state = qulacs.QuantumState(1)
            state.set_computational_basis(0b1)
            circ.update_quantum_state(state)
            denom = np.sin(theta)
            if denom==0:
                denom = 1.0e-8
            qxs.append(state.get_amplitude(0) / (1j * denom))
        return np.array(qxs)
