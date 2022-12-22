#include <cstdio>
#include <cassert>
#include "engine.h"
#include "nn.h"

void test_sanity_check(void)
{
    auto x = make_value(-4.0);
    auto z = 2 * x + 2 + x;
    auto q = cmg_relu(z) + z * x;
    auto h = cmg_relu(z * z);
    auto y = h + q + q * x;
    cmg_backward(y);

    assert(cmg_grad(x) == 46);
    assert(cmg_data(y) == -20);

    engine_free_expression(y);
}

void test_more_ops(void)
{
    auto a = make_value(-4.0);
    auto b = make_value(2.0);
    auto c = a + b;
    auto d = a * b + cmg_pow(b, 3);
    c = c + c + 1;
    c = c + 1 + c + (-a);
    d = d + d * 2 + cmg_relu(b + a);
    d = d + 3 * d + cmg_relu(b - a);
    auto e = c - d;
    auto f = cmg_pow(e, 2);
    auto g = f / 2;
    g = g + 10.0 / f;
    cmg_backward(g);

    assert(fabs(cmg_data(g) - 24.70408163265306) < 1e-3);
    assert(fabs(cmg_grad(a) - 138.83381924198252) < 1e-3);
    assert(fabs(cmg_grad(b) - 645.5772594752186) < 1e-3);

    engine_free_expression(g);
}

int main(int argc, char* arv[])
{
    engine_init();
    test_sanity_check();
    test_more_ops();
    engine_free();

    return 0;
}
