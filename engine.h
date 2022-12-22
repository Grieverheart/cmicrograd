#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

// TODO:
//     * How should we really free subexpressions? When e.g. doing
//       nn_mlp_forward(), new Values are created which refer to the MLP
//       parameters. If I were to free the expression, it would also free the
//       MLP parameters.
//     * It might be nice to be able to recognize disjoint graphs and be able
//       to walk them. This can e.g. be done by keeping an array with graph
//       ids of each Value. Furthermore to make things not slow, you need to
//       keep a list of Value ids belonging to a graph. It's generally
//       complicated and I'm trying to avoid that for a single header library.
//     * At some point, as the graphs grow, it might be better to switch to
//       hashmaps.

// Possible options for memory management:
//
// 1) Add a make_parameter() call which will mark the created values as such
//    and not free them when doing engine_free_computation(), but instead the
//    user will have to do that explicitly.
// 2) Ideally we'd like to do something like engine_push_computation() and
//    engine_pop_computation(). Then we'd probably need a variable telling us
//    on which computational graph id a value is on. Everything on that
//    computational graph will then be free'd on popping.

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
    size_t* data;
} Queue;

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
    size_t current_computation;
    Computation computation_stack[100];
} Engine;

static Engine _engine;

inline float sqr(float x)
{
    return x * x;
}

void queue_append(Queue* queue, size_t value)
{
    queue->data[queue->tail % queue->size] = value;
    queue->tail = queue->tail + 1;
}

size_t queue_pop(Queue* queue)
{
    size_t value = queue->data[queue->head];
    queue->head = queue->head + 1;
    if(queue->head == queue->size)
    {
        queue->head = 0;
        queue->tail -= queue->size;
    }
    return value;
}

size_t queue_pop_back(Queue* queue)
{
    if(queue->tail == 0)
    {
        queue->tail += queue->size;
        queue->head += queue->size;
    }
    else --queue->tail;

    size_t value = queue->data[queue->tail];
    return value;
}

void queue_create(Queue* queue, size_t size)
{
    queue->head = 0;
    queue->tail = 0;
    queue->size = size;
    queue->data = (size_t*)malloc(size * sizeof(size_t));
}

void queue_resize(Queue* queue, size_t size)
{
    queue->size = size;
    queue->data = (size_t*)realloc((void*)queue->data, size * sizeof(size_t));
}

void queue_free(Queue* queue)
{
    free(queue->data);
}

void val_print(Value val)
{
    assert(val.cid <= _engine.current_computation);
    float data = _engine.computation_stack[val.cid].data[val.id];
    float grad = _engine.computation_stack[val.cid].grad[val.id];
    int op = _engine.computation_stack[val.cid].op[val.id];
    printf("Value(data=%f, grad=%f, op=%s)\n", data, grad, OP_LABELS[(int)op]);
}

Value _make_value(float data, Value* op_ids, int op)
{
    size_t cid = _engine.current_computation;
    Computation* computation = &_engine.computation_stack[cid];

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

Value make_value(float data)
{
    return _make_value(data, NULL, OP_NOP);
}

Value val_add(Value a, Value b)
{
    Value op_ids[2] = {a, b};
    float data_a = _engine.computation_stack[a.cid].data[a.id];
    float data_b = _engine.computation_stack[b.cid].data[b.id];
    return _make_value(data_a + data_b, op_ids, OP_ADD);
}

Value val_sub(Value a, Value b)
{
    Value op_ids[2] = {a, b};
    float data_a = _engine.computation_stack[a.cid].data[a.id];
    float data_b = _engine.computation_stack[b.cid].data[b.id];
    return _make_value(data_a - data_b, op_ids, OP_SUB);
}

Value val_mul(Value a, Value b)
{
    Value op_ids[2] = {a, b};
    float data_a = _engine.computation_stack[a.cid].data[a.id];
    float data_b = _engine.computation_stack[b.cid].data[b.id];
    return _make_value(data_a * data_b, op_ids, OP_MUL);
}

Value val_div(Value a, Value b)
{
    Value op_ids[2] = {a, b};
    float data_a = _engine.computation_stack[a.cid].data[a.id];
    float data_b = _engine.computation_stack[b.cid].data[b.id];
    return _make_value(data_a / data_b, op_ids, OP_DIV);
}

Value val_pow(Value a, Value b)
{
    Value op_ids[2] = {a, b};
    float data_a = _engine.computation_stack[a.cid].data[a.id];
    float data_b = _engine.computation_stack[b.cid].data[b.id];
    return _make_value(pow(data_a, data_b), op_ids, OP_POW);
}

Value val_relu(Value a)
{
    Value dummy = {-1,-1};
    Value op_ids[2] = {a, dummy};
    float data_a = _engine.computation_stack[a.cid].data[a.id];
    return _make_value(data_a < 0.0 ? 0.0: data_a, op_ids, OP_RELU);
}

Value val_neg(Value a)
{
    return val_mul(a, make_value(-1.0));
}

float val_data(Value a)
{
    assert(a.cid == _engine.current_computation);
    return _engine.computation_stack[a.cid].data[a.id];
}

float val_grad(Value a)
{
    assert(a.cid == _engine.current_computation);
    return _engine.computation_stack[a.cid].grad[a.id];
}

void _backward(size_t id, size_t cid)
{
    Computation* computation = &_engine.computation_stack[cid];
    const Value* op_ids = computation->op_ids + 2 * id;
    float val_grad = computation->grad[id];
    switch(computation->op[id])
    {
        case OP_ADD:
        {
            _engine.computation_stack[op_ids[0].cid].grad[op_ids[0].id] += val_grad;
            _engine.computation_stack[op_ids[1].cid].grad[op_ids[1].id] += val_grad;
        }
        break;

        case OP_SUB:
        {
            _engine.computation_stack[op_ids[0].cid].grad[op_ids[0].id] += val_grad;
            _engine.computation_stack[op_ids[1].cid].grad[op_ids[1].id] -= val_grad;
        }
        break;

        case OP_MUL:
        {
            float data_a = _engine.computation_stack[op_ids[0].cid].data[op_ids[0].id];
            float data_b = _engine.computation_stack[op_ids[1].cid].data[op_ids[1].id];
            _engine.computation_stack[op_ids[0].cid].grad[op_ids[0].id] += data_b * val_grad;
            _engine.computation_stack[op_ids[1].cid].grad[op_ids[1].id] += data_a * val_grad;
        }
        break;

        case OP_DIV:
        {
            float data_a = _engine.computation_stack[op_ids[0].cid].data[op_ids[0].id];
            float data_b = _engine.computation_stack[op_ids[1].cid].data[op_ids[1].id];
            _engine.computation_stack[op_ids[0].cid].grad[op_ids[0].id] += (1.0 / data_b) * val_grad;
            _engine.computation_stack[op_ids[1].cid].grad[op_ids[1].id] -= (data_a / sqr(data_b)) * val_grad;
        }
        break;

        case OP_POW:
        {
            float data_a = _engine.computation_stack[op_ids[0].cid].data[op_ids[0].id];
            float data_b = _engine.computation_stack[op_ids[1].cid].data[op_ids[1].id];
            _engine.computation_stack[op_ids[0].cid].grad[op_ids[0].id] += (data_b * pow(data_a, data_b - 1.0)) * val_grad;
            _engine.computation_stack[op_ids[1].cid].grad[op_ids[1].id] += pow(data_a, data_b) * log(data_a) * val_grad;
        }
        break;

        case OP_RELU:
        {
            float data_a = _engine.computation_stack[op_ids[0].cid].data[op_ids[0].id];
            _engine.computation_stack[op_ids[0].cid].grad[op_ids[0].id] += (data_a > 0.0) * val_grad;
        }
        break;

        default:
        {
        }
        break;
    }
}

#if 0
void val_backward(Value val)
{
    Queue queue;
    queue_create(&queue, _engine.max_id + 1);
    int* in_degree = (int*) calloc((_engine.max_id + 1), sizeof(int));
    bool* visited = (bool*) calloc((_engine.max_id + 1), 1);

    queue_append(&queue, val.id);
    visited[val.id] = true;
    while(queue.head != queue.tail)
    {
        size_t cid = queue_pop(&queue);
        for(size_t ci = 0; ci < 2; ++ci)
        {
            size_t child_cid = _engine.op_ids[2*cid+ci];
            if(child_cid != (size_t)(-1))
            {
                if(!visited[child_cid])
                {
                    queue_append(&queue, child_cid);
                    visited[child_cid] = true;
                }
                ++in_degree[child_cid];
            }
        }
    }
    free(visited);

    _engine.grad[val.id] = 1.0;

    queue_append(&queue, val.id);
    while(queue.head != queue.tail)
    {
        size_t cid = queue_pop(&queue);
        _backward(cid);

        for(size_t ci = 0; ci < 2; ++ci)
        {
            size_t child_cid = _engine.op_ids[2*cid+ci];
            if((child_cid != (size_t)(-1)) && (--in_degree[child_cid] == 0))
                queue_append(&queue, child_cid);
        }
    }

    free(in_degree);
    queue_free(&queue);
}
#else
void build_topo(size_t id, size_t cid, bool** visited, Queue* id_queue, Queue* cid_queue)
{
    if(visited[cid][id])
        return;

    visited[cid][id] = true;
    for(size_t ci = 0; ci < 2; ++ci)
    {
        Value child = _engine.computation_stack[cid].op_ids[2*id+ci];
        if(child.id != (size_t)(-1))
        {
            build_topo(child.id, child.cid, visited, id_queue, cid_queue);
        }
    }
    queue_append(id_queue, id);
    queue_append(cid_queue, cid);
}

void val_backward(Value val)
{
    size_t num_values = 0;
    bool** visited = (bool**) malloc((_engine.current_computation + 1) * sizeof(bool*));
    for(size_t i = 0; i <= _engine.current_computation; ++i)
    {
        visited[i] = (bool*) calloc(_engine.computation_stack[i].num_values, 1);
        num_values += _engine.computation_stack[i].num_values;
    }

    Queue id_queue, cid_queue;
    queue_create(&id_queue, num_values);
    queue_create(&cid_queue, num_values);

    build_topo(val.id, val.cid, visited, &id_queue, &cid_queue);

    for(size_t i = 0; i <= _engine.current_computation; ++i)
        free(visited[i]);
    free(visited);

    _engine.computation_stack[val.cid].grad[val.id] = 1.0;
    while(id_queue.head != id_queue.tail)
    {
        size_t current_id = queue_pop_back(&id_queue);
        size_t current_cid = queue_pop_back(&cid_queue);
        _backward(current_id, current_cid);
    }

    queue_free(&id_queue);
    queue_free(&cid_queue);
}
#endif

void engine_computation_init(size_t id)
{
    Computation* computation = &_engine.computation_stack[id];

    computation->num_values  = 0;
    computation->size_values = 10;
    computation->data   = (float*) malloc(computation->size_values * sizeof(float));
    computation->grad   = (float*) malloc(computation->size_values * sizeof(float));
    computation->op     = (int*) malloc(computation->size_values * sizeof(int));
    computation->op_ids = (Value*) malloc(2 * computation->size_values * sizeof(Value));
}

void engine_init(void)
{
    _engine.current_computation = 0;
    engine_computation_init(0);
}

void engine_computation_free(size_t id)
{
    assert(_engine.current_computation == 0);
    Computation* computation = &_engine.computation_stack[id];
    free(computation->data);
    free(computation->grad);
    free(computation->op);
    free(computation->op_ids);
}

void engine_free(void)
{
    for(size_t i = 0; i <= _engine.current_computation; ++i)
        engine_computation_free(i);
}

void engine_computation_push(void)
{
    engine_computation_init(++_engine.current_computation);
}

void engine_computation_pop(void)
{
    engine_computation_free(_engine.current_computation--);
}

#ifdef __cplusplus
Value operator+(Value a, Value b)
{
    return val_add(a, b);
}
Value operator+(Value a, float b)
{
    return val_add(a, make_value(b));
}
Value operator+(float a, Value b)
{
    return val_add(make_value(a), b);
}

Value operator*(Value a, Value b)
{
    return val_mul(a, b);
}
Value operator*(Value a, float b)
{
    return val_mul(a, make_value(b));
}
Value operator*(float a, Value b)
{
    return val_mul(make_value(a), b);
}

Value operator-(Value a)
{
    return val_neg(a);
}
Value operator-(Value a, Value b)
{
    return val_sub(a, b);
}
Value operator-(Value a, float b)
{
    return val_sub(a, make_value(b));
}
Value operator-(float a, Value b)
{
    return val_sub(make_value(a), b);
}

Value val_pow(Value a, float b)
{
    return val_pow(a, make_value(b));
}

Value operator/(Value a, Value b)
{
    return val_div(a, b);
}
Value operator/(Value a, float b)
{
    return val_div(a, make_value(b));
}
Value operator/(float a, Value b)
{
    return val_div(make_value(a), b);
}
#endif
