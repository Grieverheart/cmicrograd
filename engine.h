#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

// TODO:
//     * Finish MLP.
//     * Do SGD test.
//     * At some point, as the graphs grow, it might be better to switch to
//       hashmaps.

enum Operation
{
    OP_NOP,
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_POW,
    OP_RELU
};

static const char* OP_LABELS[] =
{
    "NOP",
    "ADD",
    "SUB",
    "MUL",
    "DIV",
    "POW",
    "RELU"
};

typedef struct
{
    size_t id;
    size_t cid;
} Value;

typedef struct
{
    size_t head;
    size_t tail;
    size_t size;
    Value* data;
} ValueQueue;

typedef struct
{
    float*  data;
    float*  grad;
    int*    op;
    Value*  op_ids;
    size_t  num_values;
    size_t  size_values;
} Computation;

typedef struct
{
    size_t stack_id;
    Computation stack[100];
} Engine;

static Engine _engine;

inline float sqr(float x)
{
    return x * x;
}

void value_queue_push(ValueQueue* queue, Value value)
{
    queue->data[queue->tail % queue->size] = value;
    queue->tail = queue->tail + 1;
}

Value value_queue_pop(ValueQueue* queue)
{
    Value value = queue->data[queue->head];
    queue->head = queue->head + 1;
    if(queue->head == queue->size)
    {
        queue->head = 0;
        queue->tail -= queue->size;
    }
    return value;
}

Value value_queue_pop_back(ValueQueue* queue)
{
    if(queue->tail == 0)
    {
        queue->tail += queue->size;
        queue->head += queue->size;
    }
    else --queue->tail;

    Value value = queue->data[queue->tail];
    return value;
}

void value_queue_create(ValueQueue* queue, size_t size)
{
    queue->head = 0;
    queue->tail = 0;
    queue->size = size;
    queue->data = (Value*)malloc(size * sizeof(Value));
}

void value_queue_free(ValueQueue* queue)
{
    free(queue->data);
}

void cmg_print(Value val)
{
    assert(val.cid <= _engine.stack_id);
    float data = _engine.stack[val.cid].data[val.id];
    float grad = _engine.stack[val.cid].grad[val.id];
    int op = _engine.stack[val.cid].op[val.id];
    printf("Value(data=%f, grad=%f, op=%s)\n", data, grad, OP_LABELS[(int)op]);
}

Value _cmg_value(float data, Value* op_ids, int op)
{
    size_t cid = _engine.stack_id;
    Computation* computation = &_engine.stack[cid];

    if(computation->num_values == computation->size_values)
    {
        computation->size_values *= 3;
        computation->size_values /= 2;
        computation->data      = (float*) realloc((void*)computation->data, computation->size_values * sizeof(float));
        computation->grad      = (float*) realloc((void*)computation->grad, computation->size_values * sizeof(float));
        computation->op        = (int*) realloc((void*)computation->op, computation->size_values * sizeof(int));
        computation->op_ids    = (Value*) realloc((void*)computation->op_ids, 2 * computation->size_values * sizeof(Value));
    }

    size_t id = computation->num_values++;

    Value dummy = {-1,-1};

    computation->data[id]       = data;
    computation->grad[id]       = 0.0f;
    computation->op[id]         = op;
    computation->op_ids[2*id+0] = op_ids? op_ids[0]: dummy;
    computation->op_ids[2*id+1] = op_ids? op_ids[1]: dummy;

    Value val = {id, cid};
    return val;
}

Value cmg_val(float data)
{
    return _cmg_value(data, NULL, OP_NOP);
}

Value cmg_add(Value a, Value b)
{
    Value op_ids[2] = {a, b};
    float data_a = _engine.stack[a.cid].data[a.id];
    float data_b = _engine.stack[b.cid].data[b.id];
    return _cmg_value(data_a + data_b, op_ids, OP_ADD);
}

Value cmg_sub(Value a, Value b)
{
    Value op_ids[2] = {a, b};
    float data_a = _engine.stack[a.cid].data[a.id];
    float data_b = _engine.stack[b.cid].data[b.id];
    return _cmg_value(data_a - data_b, op_ids, OP_SUB);
}

Value cmg_mul(Value a, Value b)
{
    Value op_ids[2] = {a, b};
    float data_a = _engine.stack[a.cid].data[a.id];
    float data_b = _engine.stack[b.cid].data[b.id];
    return _cmg_value(data_a * data_b, op_ids, OP_MUL);
}

Value cmg_div(Value a, Value b)
{
    Value op_ids[2] = {a, b};
    float data_a = _engine.stack[a.cid].data[a.id];
    float data_b = _engine.stack[b.cid].data[b.id];
    return _cmg_value(data_a / data_b, op_ids, OP_DIV);
}

Value cmg_pow(Value a, Value b)
{
    Value op_ids[2] = {a, b};
    float data_a = _engine.stack[a.cid].data[a.id];
    float data_b = _engine.stack[b.cid].data[b.id];
    return _cmg_value(pow(data_a, data_b), op_ids, OP_POW);
}

Value cmg_relu(Value a)
{
    Value dummy = {-1,-1};
    Value op_ids[2] = {a, dummy};
    float data_a = _engine.stack[a.cid].data[a.id];
    return _cmg_value(data_a < 0.0 ? 0.0: data_a, op_ids, OP_RELU);
}

Value cmg_neg(Value a)
{
    return cmg_mul(a, cmg_val(-1.0));
}

float cmg_data(Value a)
{
    assert(a.cid == _engine.stack_id);
    return _engine.stack[a.cid].data[a.id];
}

float cmg_grad(Value a)
{
    assert(a.cid == _engine.stack_id);
    return _engine.stack[a.cid].grad[a.id];
}

void _backward(Value val)
{
    size_t id = val.id;
    size_t cid = val.cid;

    Computation* computation = &_engine.stack[cid];
    const Value* op_ids = computation->op_ids + 2 * id;
    float cmg_grad = computation->grad[id];
    switch(computation->op[id])
    {
        case OP_ADD:
        {
            _engine.stack[op_ids[0].cid].grad[op_ids[0].id] += cmg_grad;
            _engine.stack[op_ids[1].cid].grad[op_ids[1].id] += cmg_grad;
        }
        break;

        case OP_SUB:
        {
            _engine.stack[op_ids[0].cid].grad[op_ids[0].id] += cmg_grad;
            _engine.stack[op_ids[1].cid].grad[op_ids[1].id] -= cmg_grad;
        }
        break;

        case OP_MUL:
        {
            float data_a = _engine.stack[op_ids[0].cid].data[op_ids[0].id];
            float data_b = _engine.stack[op_ids[1].cid].data[op_ids[1].id];
            _engine.stack[op_ids[0].cid].grad[op_ids[0].id] += data_b * cmg_grad;
            _engine.stack[op_ids[1].cid].grad[op_ids[1].id] += data_a * cmg_grad;
        }
        break;

        case OP_DIV:
        {
            float data_a = _engine.stack[op_ids[0].cid].data[op_ids[0].id];
            float data_b = _engine.stack[op_ids[1].cid].data[op_ids[1].id];
            _engine.stack[op_ids[0].cid].grad[op_ids[0].id] += (1.0 / data_b) * cmg_grad;
            _engine.stack[op_ids[1].cid].grad[op_ids[1].id] -= (data_a / sqr(data_b)) * cmg_grad;
        }
        break;

        case OP_POW:
        {
            float data_a = _engine.stack[op_ids[0].cid].data[op_ids[0].id];
            float data_b = _engine.stack[op_ids[1].cid].data[op_ids[1].id];
            _engine.stack[op_ids[0].cid].grad[op_ids[0].id] += (data_b * pow(data_a, data_b - 1.0)) * cmg_grad;
            _engine.stack[op_ids[1].cid].grad[op_ids[1].id] += pow(data_a, data_b) * log(data_a) * cmg_grad;
        }
        break;

        case OP_RELU:
        {
            float data_a = _engine.stack[op_ids[0].cid].data[op_ids[0].id];
            _engine.stack[op_ids[0].cid].grad[op_ids[0].id] += (data_a > 0.0) * cmg_grad;
        }
        break;

        default:
        {
        }
        break;
    }
}

#if 0
void cmg_backward(Value val)
{
    Queue queue;
    value_queue_create(&queue, _engine.max_id + 1);
    int* in_degree = (int*) calloc((_engine.max_id + 1), sizeof(int));
    bool* visited = (bool*) calloc((_engine.max_id + 1), 1);

    value_queue_push(&queue, val.id);
    visited[val.id] = true;
    while(queue.head != queue.tail)
    {
        size_t cid = value_queue_pop(&queue);
        for(size_t ci = 0; ci < 2; ++ci)
        {
            size_t child_cid = _engine.op_ids[2*cid+ci];
            if(child_cid != (size_t)(-1))
            {
                if(!visited[child_cid])
                {
                    value_queue_push(&queue, child_cid);
                    visited[child_cid] = true;
                }
                ++in_degree[child_cid];
            }
        }
    }
    free(visited);

    _engine.grad[val.id] = 1.0;

    value_queue_push(&queue, val.id);
    while(queue.head != queue.tail)
    {
        size_t cid = value_queue_pop(&queue);
        _backward(cid);

        for(size_t ci = 0; ci < 2; ++ci)
        {
            size_t child_cid = _engine.op_ids[2*cid+ci];
            if((child_cid != (size_t)(-1)) && (--in_degree[child_cid] == 0))
                value_queue_push(&queue, child_cid);
        }
    }

    free(in_degree);
    value_queue_free(&queue);
}
#else
void build_topo(Value val, bool** visited, ValueQueue* queue)
{
    size_t id = val.id;
    size_t cid = val.cid;

    if(visited[cid][id])
        return;

    visited[cid][id] = true;
    for(size_t ci = 0; ci < 2; ++ci)
    {
        Value child = _engine.stack[cid].op_ids[2*id+ci];
        if(child.id != (size_t)(-1))
        {
            build_topo(child, visited, queue);
        }
    }
    value_queue_push(queue, val);
}

void cmg_backward(Value val)
{
    size_t num_values = 0;
    bool** visited = (bool**) malloc((_engine.stack_id + 1) * sizeof(bool*));
    for(size_t i = 0; i <= _engine.stack_id; ++i)
    {
        visited[i] = (bool*) calloc(_engine.stack[i].num_values, 1);
        num_values += _engine.stack[i].num_values;
    }

    ValueQueue queue;
    value_queue_create(&queue, num_values);

    build_topo(val, visited, &queue);

    for(size_t i = 0; i <= _engine.stack_id; ++i)
        free(visited[i]);
    free(visited);

    _engine.stack[val.cid].grad[val.id] = 1.0;
    while(queue.head != queue.tail)
    {
        Value val = value_queue_pop_back(&queue);
        _backward(val);
    }

    value_queue_free(&queue);
}
#endif

void _engine_computation_init(size_t id)
{
    Computation* computation = &_engine.stack[id];

    computation->num_values  = 0;
    computation->size_values = 10;
    computation->data   = (float*) malloc(computation->size_values * sizeof(float));
    computation->grad   = (float*) malloc(computation->size_values * sizeof(float));
    computation->op     = (int*) malloc(computation->size_values * sizeof(int));
    computation->op_ids = (Value*) malloc(2 * computation->size_values * sizeof(Value));
}

void cmg_init(void)
{
    _engine.stack_id = 0;
    _engine_computation_init(0);
}

void _engine_computation_free(size_t id)
{
    assert(_engine.stack_id == 0);
    Computation* computation = &_engine.stack[id];
    free(computation->data);
    free(computation->grad);
    free(computation->op);
    free(computation->op_ids);
}

void cmg_free(void)
{
    for(size_t i = 0; i <= _engine.stack_id; ++i)
        _engine_computation_free(i);
}

void cmg_computation_push(void)
{
    _engine_computation_init(++_engine.stack_id);
}

void cmg_computation_pop(void)
{
    assert(_engine.stack_id > 0);
    _engine_computation_free(_engine.stack_id--);
}

#ifdef __cplusplus
Value operator+(Value a, Value b)
{
    return cmg_add(a, b);
}
Value operator+(Value a, float b)
{
    return cmg_add(a, cmg_val(b));
}
Value operator+(float a, Value b)
{
    return cmg_add(cmg_val(a), b);
}

Value operator*(Value a, Value b)
{
    return cmg_mul(a, b);
}
Value operator*(Value a, float b)
{
    return cmg_mul(a, cmg_val(b));
}
Value operator*(float a, Value b)
{
    return cmg_mul(cmg_val(a), b);
}

Value operator-(Value a)
{
    return cmg_neg(a);
}
Value operator-(Value a, Value b)
{
    return cmg_sub(a, b);
}
Value operator-(Value a, float b)
{
    return cmg_sub(a, cmg_val(b));
}
Value operator-(float a, Value b)
{
    return cmg_sub(cmg_val(a), b);
}

Value cmg_pow(Value a, float b)
{
    return cmg_pow(a, cmg_val(b));
}

Value operator/(Value a, Value b)
{
    return cmg_div(a, b);
}
Value operator/(Value a, float b)
{
    return cmg_div(a, cmg_val(b));
}
Value operator/(float a, Value b)
{
    return cmg_div(cmg_val(a), b);
}
#endif
