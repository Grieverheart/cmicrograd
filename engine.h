#include <cstdlib>
#include <cmath>

enum class Operation
{
    NOP,
    ADD,
    SUB,
    NEG,
    MUL,
    DIV,
    POW,
    RELU,
};

static const char* OP_LABELS[]
{
    "NOP",
    "ADD",
    "SUB",
    "NEG",
    "MUL",
    "DIV",
    "POW",
    "RELU",
};

struct Value
{
    size_t id;
};

struct Queue
{
    size_t head;
    size_t tail;
    size_t size;
    size_t* data;
};

struct Engine
{
    float*     data;
    float*     grad;
    Operation* op;
    size_t*    op_ids;

    Queue free_queue;

    size_t max_id;
    size_t size_values;
};

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

Value make_value(float data, size_t* op_ids = nullptr, Operation op = Operation::NOP)
{
    if(_engine.free_queue.head == _engine.free_queue.tail)
    {
        size_t size_values_old = _engine.size_values;
        _engine.size_values *= 3;
        _engine.size_values /= 2;
        _engine.data   = (float*) realloc((void*)_engine.data, _engine.size_values * sizeof(float));
        _engine.grad   = (float*) realloc((void*)_engine.grad, _engine.size_values * sizeof(float));
        _engine.op     = (Operation*) realloc((void*)_engine.op, _engine.size_values * sizeof(Operation));
        _engine.op_ids = (size_t*) realloc((void*)_engine.op_ids, 2 * _engine.size_values * sizeof(size_t));
        queue_resize(&_engine.free_queue, _engine.size_values);
        for(size_t i = size_values_old; i < _engine.size_values; ++i)
            queue_append(&_engine.free_queue, i);
    }

    size_t id = queue_pop(&_engine.free_queue);
    if(id > _engine.max_id) _engine.max_id = id;

    _engine.data[id]       = data;
    _engine.grad[id]       = 0.0f;
    _engine.op[id]         = op;
    _engine.op_ids[2*id+0] = op_ids? op_ids[0]: size_t(-1);
    _engine.op_ids[2*id+1] = op_ids? op_ids[1]: size_t(-1);

    return Value{id};
}

Value val_add(Value a, Value b)
{
    size_t op_ids[2] = {a.id, b.id};
    float data_a = _engine.data[a.id];
    float data_b = _engine.data[b.id];
    return make_value(data_a + data_b, op_ids, Operation::ADD);
}

Value val_sub(Value a, Value b)
{
    size_t op_ids[2] = {a.id, b.id};
    float data_a = _engine.data[a.id];
    float data_b = _engine.data[b.id];
    return make_value(data_a - data_b, op_ids, Operation::SUB);
}

Value val_mul(Value a, Value b)
{
    size_t op_ids[2] = {a.id, b.id};
    float data_a = _engine.data[a.id];
    float data_b = _engine.data[b.id];
    return make_value(data_a * data_b, op_ids, Operation::MUL);
}

Value val_div(Value a, Value b)
{
    size_t op_ids[2] = {a.id, b.id};
    float data_a = _engine.data[a.id];
    float data_b = _engine.data[b.id];
    return make_value(data_a / data_b, op_ids, Operation::DIV);
}

Value val_pow(Value a, Value b)
{
    size_t op_ids[2] = {a.id, b.id};
    float data_a = _engine.data[a.id];
    float data_b = _engine.data[b.id];
    return make_value(pow(data_a, data_b), op_ids, Operation::POW);
}

Value val_relu(Value a)
{
    size_t op_ids[2] = {a.id, size_t(-1)};
    float data_a = _engine.data[a.id];
    return make_value(data_a < 0.0 ? 0.0: data_a, op_ids, Operation::RELU);
}

float val_data(Value a)
{
    return _engine.data[a.id];
}

float val_grad(Value a)
{
    return _engine.grad[a.id];
}

void _backward(size_t id)
{
    const size_t* op_ids = _engine.op_ids + 2 * id;
    float val_grad = _engine.grad[id];
    switch(_engine.op[id])
    {
        case Operation::ADD:
        {
            _engine.grad[op_ids[0]] += val_grad;
            _engine.grad[op_ids[1]] += val_grad;
        }
        break;

        case Operation::SUB:
        {
            _engine.grad[op_ids[0]] += val_grad;
            _engine.grad[op_ids[1]] -= val_grad;
        }
        break;

        case Operation::NEG:
        {
            _engine.grad[op_ids[0]] -= val_grad;
        }
        break;

        case Operation::MUL:
        {
            float data_a = _engine.data[op_ids[0]];
            float data_b = _engine.data[op_ids[1]];
            _engine.grad[op_ids[0]] += data_b * val_grad;
            _engine.grad[op_ids[1]] += data_a * val_grad;
        }
        break;

        case Operation::DIV:
        {
            float data_a = _engine.data[op_ids[0]];
            float data_b = _engine.data[op_ids[1]];
            _engine.grad[op_ids[0]] += (1.0 / data_b) * val_grad;
            _engine.grad[op_ids[1]] -= (data_a / sqr(data_b)) * val_grad;
        }
        break;

        case Operation::POW:
        {
            float data_a = _engine.data[op_ids[0]];
            float data_b = _engine.data[op_ids[1]];
            _engine.grad[op_ids[0]] += (data_b * pow(data_a, data_b - 1.0)) * val_grad;
            _engine.grad[op_ids[1]] += pow(data_a, data_b) * log(data_a) * val_grad;
        }
        break;

        case Operation::RELU:
        {
            float data_a = _engine.data[op_ids[0]];
            _engine.grad[op_ids[0]] += (data_a > 0.0) * val_grad;
        }
        break;

        default:
        {
        }
        break;
    }
}

void val_print(Value val)
{
    float data = _engine.data[val.id];
    float grad = _engine.grad[val.id];
    Operation op = _engine.op[val.id];
    printf("Value(data=%f, grad=%f, op=%s)\n", data, grad, OP_LABELS[(int)op]);
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
            if(child_cid != size_t(-1))
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
            if((child_cid != size_t(-1)) && (--in_degree[child_cid] == 0))
                queue_append(&queue, child_cid);
        }
    }

    free(in_degree);
    queue_free(&queue);
}
#else
void build_topo(size_t cid, bool* visited, Queue* queue)
{
    if(visited[cid])
        return;

    visited[cid] = true;
    for(size_t ci = 0; ci < 2; ++ci)
    {
        size_t child_cid = _engine.op_ids[2*cid+ci];
        if(child_cid != size_t(-1))
        {
            build_topo(child_cid, visited, queue);
        }
    }
    queue_append(queue, cid);
}

void val_backward(Value val)
{
    Queue queue;
    queue_create(&queue, _engine.max_id + 1);
    bool* visited = (bool*) calloc((_engine.max_id + 1), 1);
    build_topo(val.id, visited, &queue);
    free(visited);

    _engine.grad[val.id] = 1.0;
    while(queue.head != queue.tail)
    {
        size_t cid = queue_pop_back(&queue);
        _backward(cid);
    }

    queue_free(&queue);
}
#endif

void engine_init(void)
{
    _engine.max_id  = 0;
    _engine.size_values = 10;
    _engine.data   = (float*) malloc(_engine.size_values * sizeof(float));
    _engine.grad   = (float*) malloc(_engine.size_values * sizeof(float));
    _engine.op     = (Operation*) malloc(_engine.size_values * sizeof(Operation));
    _engine.op_ids = (size_t*) malloc(2 * _engine.size_values * sizeof(size_t));

    queue_create(&_engine.free_queue, _engine.size_values);
    for(size_t i = 0; i < _engine.size_values; ++i)
        queue_append(&_engine.free_queue, i);
}

void engine_free(void)
{
    free(_engine.data);
    free(_engine.grad);
    free(_engine.op);
    free(_engine.op_ids);
    queue_free(&_engine.free_queue);
}

void engine_free_expression(Value val)
{
    Queue queue;
    queue_create(&queue, _engine.max_id + 1);
    bool* visited = (bool*) calloc((_engine.max_id + 1), 1);

    _engine.grad[val.id] = 1.0;

    while(queue.head != queue.tail)
    {
        size_t cid = queue_pop(&queue);
        queue_append(&_engine.free_queue, cid);
        if(!visited[cid])
        {
            visited[cid] = true;
            for(size_t ci = 0; ci < 2; ++ci)
            {
                if(_engine.op_ids[2*cid+ci] != size_t(-1))
                    queue_append(&queue, _engine.op_ids[2*cid+ci]);
            }
        }
    }

    free(visited);
    queue_free(&queue);
}

