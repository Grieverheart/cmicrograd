#include <stdio.h>
#include <assert.h>
#include "engine.h"
#include "nn.h"

void test_sanity_check(void)
{
    engine_computation_push();

    Value x = make_value(-4.0);
    Value z = val_add(val_add(val_mul(make_value(2), x), make_value(2)), x);
    Value q = val_add(val_relu(z), val_mul(z, x));
    Value h = val_relu(val_mul(z, z));
    Value y = val_add(val_add(h, q), val_mul(q, x));
    val_backward(y);

    assert(val_grad(x) == 46);
    assert(val_data(y) == -20);

    engine_computation_pop();
}

void test_more_ops(void)
{
    engine_computation_push();

    // @note: Perhaps a possibility to make a parser for computations similar
    // to __asm__.
    //Computation computation = make_computation(
    //    "%0 = -4.0;"
    //    "%1 = 2.0;"
    //    "c = %0 + %1;"
    //    "d = %0 * %1 + %1**3;"
    //    "c = c + c + 1;"
    //    "c = c + 1 + c + (-%0);"
    //    "d = d + d * 2 + relu(%1 + %0);"
    //    "d = d + 3 * d + relu(%1 - %0);"
    //    "e = c - d;"
    //    "f = e**2;"
    //    "g = f / 2;"
    //    "g = g + 10.0 / f;",
    //    a, b, g
    //);

    Value a = make_value(-4.0);
    Value b = make_value(2.0);
    Value c = val_add(a, b);
    Value d = val_add(val_mul(a, b), val_pow(b, make_value(3)));
    c = val_add(c, val_add(c, make_value(1)));
    c = val_add(c, val_add(make_value(1), val_sub(c, a)));
    d = val_add(d, val_add(val_mul(d, make_value(2)), val_relu(val_add(b, a))));
    d = val_add(d, val_add(val_mul(make_value(3), d), val_relu(val_sub(b, a))));
    Value e = val_sub(c, d);
    Value f = val_pow(e, make_value(2));
    Value g = val_div(f, make_value(2));
    g = val_add(g, val_div(make_value(10.0), f));
    val_backward(g);

    assert(fabs(val_data(g) - 24.70408163265306) < 1e-3);
    assert(fabs(val_grad(a) - 138.83381924198252) < 1e-3);
    assert(fabs(val_grad(b) - 645.5772594752186) < 1e-3);

    engine_computation_pop();
}

int main(int argc, char* arv[])
{
    engine_init();
    test_sanity_check();
    test_more_ops();

    MLP mlp;
    size_t sizes[] = {784, 30, 10};
    nn_mlp_create(&mlp, sizes, 3);
    {
        engine_computation_push();
        {
            Value x[784];
            for(size_t i = 0; i < 784; ++i)
                x[i] = make_value(1.0/784.0);
            Value* y = nn_mlp_forward(&mlp, x);
            for(size_t i = 0; i < 10; ++i)
                val_backward(y[i]);
            free(y);
        }
        engine_computation_pop();

        size_t num_params = 0;
        Value* params = nn_mlp_params(&mlp, &num_params);
        printf("%lu\n", num_params);

        for(size_t i = 0; i < 10; ++i)
            printf("%f\n", val_grad(params[i]));

        free(params);
    }
    nn_mlp_free(&mlp);

    engine_free();

    return 0;
}
