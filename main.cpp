#include <cstdio>
#include "engine.h"

void test_sanity_check(void)
{
    auto x = make_value(-4.0);
    auto z = val_add(val_add(val_mul(make_value(2), x), make_value(2)), x);
    auto q = val_add(val_relu(z), val_mul(z, x));
    auto h = val_relu(val_mul(z, z));
    auto y = val_add(val_add(h, q), val_mul(q, x));
    val_backward(y);
    val_print(x);
    val_print(y);

    //x = torch.Tensor([-4.0]).double()
    //x.requires_grad = True
    //z = 2 * x + 2 + x
    //q = z.relu() + z * x
    //h = (z * z).relu()
    //y = h + q + q * x
    //y.backward()
    //xpt, ypt = x, y

    //# forward pass went well
    //assert ymg.data == ypt.data.item()
    //# backward pass went well
    //assert xmg.grad == xpt.grad.item()
}

int main(int argc, char* arv[])
{
    engine_init();
    test_sanity_check();
    engine_free();

    return 0;
}
