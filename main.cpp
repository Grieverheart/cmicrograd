#include <cstdio>
#include <cassert>
#include "engine.h"

void test_sanity_check(void)
{
    auto x = make_value(-4.0);
    auto z = val_add(val_add(val_mul(make_value(2), x), make_value(2)), x);
    auto q = val_add(val_relu(z), val_mul(z, x));
    auto h = val_relu(val_mul(z, z));
    auto y = val_add(val_add(h, q), val_mul(q, x));
    val_backward(y);

    assert(val_grad(x) == 46);
    assert(val_data(y) == -20);

    engine_free_expression(y);
}

void test_more_ops(void)
{
    auto a = make_value(-4.0);
    auto b = make_value(2.0);
    auto c = val_add(a, b);
    auto d = val_add(val_mul(a, b), val_pow(b, make_value(3)));
    c = val_add(c, val_add(c, make_value(1)));
    c = val_add(c, val_add(make_value(1), val_sub(c, a)));
    d = val_add(d, val_add(val_mul(d, make_value(2)), val_relu(val_add(b, a))));
    d = val_add(d, val_add(val_mul(make_value(3), d), val_relu(val_sub(b, a))));
    auto e = val_sub(c, d);
    auto f = val_pow(e, make_value(2));
    auto g = val_div(f, make_value(2));
    g = val_add(g, val_div(make_value(10.0), f));
    val_backward(g);

    assert(fabs(val_data(g) - 24.70408163265306) < 1e-3);
    assert(fabs(val_grad(a) - 138.83381924198252) < 1e-3);
    assert(fabs(val_grad(b) - 645.5772594752186) < 1e-3);

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
