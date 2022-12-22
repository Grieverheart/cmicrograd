#include <stdio.h>
#include <assert.h>
#include "engine.h"
#include "nn.h"

void test_sanity_check(void)
{
    cmg_computation_push();

    Value x = cmg_val(-4.0);
    Value z = cmg_add(cmg_add(cmg_mul(cmg_val(2), x), cmg_val(2)), x);
    Value q = cmg_add(cmg_relu(z), cmg_mul(z, x));
    Value h = cmg_relu(cmg_mul(z, z));
    Value y = cmg_add(cmg_add(h, q), cmg_mul(q, x));
    cmg_backward(y);

    assert(cmg_grad(x) == 46);
    assert(cmg_data(y) == -20);

    cmg_computation_pop();
}

void test_more_ops(void)
{
    cmg_computation_push();

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

    Value a = cmg_val(-4.0);
    Value b = cmg_val(2.0);
    Value c = cmg_add(a, b);
    Value d = cmg_add(cmg_mul(a, b), cmg_pow(b, cmg_val(3)));
    c = cmg_add(c, cmg_add(c, cmg_val(1)));
    c = cmg_add(c, cmg_add(cmg_val(1), cmg_sub(c, a)));
    d = cmg_add(d, cmg_add(cmg_mul(d, cmg_val(2)), cmg_relu(cmg_add(b, a))));
    d = cmg_add(d, cmg_add(cmg_mul(cmg_val(3), d), cmg_relu(cmg_sub(b, a))));
    Value e = cmg_sub(c, d);
    Value f = cmg_pow(e, cmg_val(2));
    Value g = cmg_div(f, cmg_val(2));
    g = cmg_add(g, cmg_div(cmg_val(10.0), f));
    cmg_backward(g);

    assert(fabs(cmg_data(g) - 24.70408163265306) < 1e-3);
    assert(fabs(cmg_grad(a) - 138.83381924198252) < 1e-3);
    assert(fabs(cmg_grad(b) - 645.5772594752186) < 1e-3);

    cmg_computation_pop();
}

int main(int argc, char* arv[])
{
    cmg_init();
    test_sanity_check();
    test_more_ops();

    MLP mlp;
    size_t sizes[] = {784, 30, 10};
    cmg_mlp_create(&mlp, sizes, 3);
    {
        cmg_computation_push();
        {
            Value x[784];
            for(size_t i = 0; i < 784; ++i)
                x[i] = cmg_val(1.0/784.0);
            Value* y = cmg_mlp_forward(&mlp, x);
            for(size_t i = 0; i < 10; ++i)
                cmg_backward(y[i]);
            free(y);
        }
        cmg_computation_pop();

        size_t num_params = 0;
        Value* params = cmg_mlp_params(&mlp, &num_params);
        printf("%lu\n", num_params);

        for(size_t i = 0; i < 10; ++i)
            printf("%f\n", cmg_grad(params[i]));

        free(params);
    }
    cmg_mlp_free(&mlp);

    cmg_free();

    return 0;
}
