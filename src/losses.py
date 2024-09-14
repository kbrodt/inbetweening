import torch
import torch.nn as nn


@torch.jit.script
def mat_from_v(v):#, dim=-1):
    dim = -1
    _v0 = v[:, 0]
    A = torch.cat(
        [
            (v[:, 1] - _v0).unsqueeze(dim),  # [N, 2, 1]
            (v[:, 2] - _v0).unsqueeze(dim),  # [N, 2, 1]
        ],
        dim=dim,
    )

    return A


@torch.jit.script
def det2x2(A):
    det = A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0]

    return det


@torch.jit.script
def inv2x2(A):
    Ainv = torch.stack(
        [
            torch.stack(
                [
                    A[:, 1, 1], -A[:, 0, 1],
                ],
                dim=1,
            ),
            torch.stack(
                [
                    -A[:, 1, 0], A[:, 0, 0],
                ],
                dim=1,
            ),
        ],
        dim=1,
    )

    return Ainv


@torch.jit.script
def invdet2x2(A):
    det = det2x2(A)
    Ainv = inv2x2(A)
    Ainv /= det.unsqueeze(-1).unsqueeze(-1)

    return Ainv


def calc_jac(t_old, t_new, dim=-1):
    A = mat_from_v(t_old)# dim=dim)
    det = det2x2(A)
    a = 0.5 * torch.abs(det)
    Ainv = inv2x2(A)
    Ainv /= det.unsqueeze(-1).unsqueeze(-1)
    B = mat_from_v(t_new)#, dim=dim)
    J = B @ Ainv

    return J, a


@torch.jit.script
def calc_areas(v):
    v = v - v[:, [0]]
    v = v[:, 1:]
    a = v[:, 0, 0] * v[:, 1, 1] - v[:, 0, 1] * v[:, 1, 0]
    a = torch.abs(a) / 2

    return a


def calc_symmetric_dirichlet(V_old, V_new, s=1e-2, reduce=False):
    # print(V_old.shape, V_new.shape)
    J, a = calc_jac(V_old, V_new)
    a /= a.sum(0)
    Jinv = torch.linalg.inv(J)
    # a2 = calc_areas(V_old)
    # a2 = torch.linalg.det(Jinv) / 2
    # print(abs(a - a2).max())
    # a = a.unsqueeze(-1).unsqueeze(-1)
    loss = (
        J ** 2 + Jinv ** 2
        #torch.abs(J) + torch.abs(Jinv)
        # - 2 * torch.tile(torch.eye(2), (len(J), 1)).view(len(J), 2, 2)
    )
    loss = torch.sum(loss, dim=(-2, -1))
    #loss = torch.exp(s * loss)
    #loss = a * loss
    #loss = loss
    if reduce:
        #assert np.allclose(a.sum(0).item(), 1), a.sum(0)
        loss = torch.sum(loss, dim=0)

    # loss = torch.linalg.matrix_norm(J) + torch.linalg.matrix_norm(Jinv)

    # JT = torch.transpose(J, -2, -1)  # J.mT
    # loss = vmap(torch.trace)(JT @ J) / torch.linalg.det(J)
    # loss = torch.linalg.matrix_norm(torch.log(JT @ J)) ** 2

    return loss#.sum(0)


@torch.jit.script
def sym_dir(Ainv, B, a):
    J = B @ Ainv

    Jinv = invdet2x2(J)

    loss = J * J + Jinv * Jinv
    #loss = torch.abs(J) + torch.abs(Jinv)
    loss = torch.sum(loss, dim=(-2, -1))
    #loss = torch.exp(s * loss)
    loss = a * loss
    loss = torch.sum(loss, dim=0)

    return loss


@torch.jit.script
def sym_dir2(Ainv, B, a):
    J = B @ Ainv
    j11 = J[:, 0, 0]
    j12 = J[:, 0, 1]
    j21 = J[:, 1, 0]
    j22 = J[:, 1, 1]
    abcd2 = j11 ** 2 + j12 ** 2 + j21 ** 2 + j22 ** 2
    det = j11 * j22 - j12 * j21
    det = det * det
    det = 1 / det
    det = det + 1
    loss = abcd2 * det
    loss = a * loss
    loss = torch.sum(loss, dim=0)

    return loss


class SymDir(nn.Module):
    def __init__(self, V_old, dim=-1):
        super().__init__()

        self.dim = dim
        self.A = mat_from_v(V_old)#, dim=self.dim)
        det = det2x2(self.A)
        self.a = 0.5 * torch.abs(det)
        self.a /= self.a.sum(0)
        #self.asum = self.a.sum(0)
        self.Ainv = inv2x2(self.A)
        self.Ainv /= det.unsqueeze(-1).unsqueeze(-1)

    #@profile
    def forward(self, y):
        B = mat_from_v(y)#, dim=self.dim)
        #loss = sym_dir(self.Ainv, B, self.a)
        loss = sym_dir2(self.Ainv, B, self.a)

        #assert torch.isclose(loss, loss2), (loss, loss2)

        #J = B @ self.Ainv

        #Jinv = invdet2x2(J)

        #loss = J * J + Jinv * Jinv
        #loss = torch.sum(loss, dim=(-2, -1))
        ##loss = torch.exp(s * loss)
        #loss = self.a * loss
        #loss = torch.sum(loss, dim=0)

        return loss


class DAD(nn.Module):
    def __init__(self, A):
        super().__init__()

        self.A = A

    def forward(self, x):
        loss = torch.trace(x.T @ self.A @ x)

        return loss
