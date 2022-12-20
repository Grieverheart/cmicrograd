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
    float data;
    float grad;
    Operation op;
    size_t op_ids[2];
};

struct Engine
{
    Value* values;
    size_t num_values;
    size_t size_values;
};

static Engine _engine;

inline float sqr(float x)
{
    return x * x;
}

Value make_value(float data, size_t* op_ids = nullptr, Operation op = Operation::NOP)
{
    if(_engine.num_values == _engine.size_values)
    {
        _engine.size_values *= 3;
        _engine.size_values /= 2;
        _engine.values = (Value*) realloc((void*)_engine.values, _engine.size_values * sizeof(Value));
    }

    Value* val = &_engine.values[_engine.num_values++];

    val->id          = _engine.num_values - 1;
    val->data        = data;
    val->grad        = 0.0f;
    val->op          = op;
    val->op_ids[0] = op_ids? op_ids[0]: size_t(-1);
    val->op_ids[1] = op_ids? op_ids[1]: size_t(-1);

    return *val;
}

Value val_add(Value a, Value b)
{
    size_t op_ids[2] = {a.id, b.id};
    auto val = make_value(a.data + b.data, op_ids, Operation::ADD);
    return val;
}

Value val_mul(Value a, Value b)
{
    size_t op_ids[2] = {a.id, b.id};
    auto val = make_value(a.data * b.data, op_ids, Operation::MUL);
    return val;
}

Value val_relu(Value a)
{
    size_t op_ids[2] = {a.id, size_t(-1)};
    auto val = make_value(a.data > 0.0 ? a.data: 0.0, op_ids, Operation::RELU);
    return val;
}

void _backward(Value* val)
{
    switch(val->op)
    {
        case Operation::ADD:
        {
            auto a = &_engine.values[val->op_ids[0]];
            auto b = &_engine.values[val->op_ids[1]];
            a->grad += val->grad;
            b->grad += val->grad;
        }
        break;

        case Operation::SUB:
        {
            auto a = &_engine.values[val->op_ids[0]];
            auto b = &_engine.values[val->op_ids[1]];
            a->grad -= val->grad;
            b->grad -= val->grad;
        }
        break;

        case Operation::NEG:
        {
            auto a = &_engine.values[val->op_ids[0]];
            a->grad -= val->grad;
        }
        break;

        case Operation::MUL:
        {
            auto a = &_engine.values[val->op_ids[0]];
            auto b = &_engine.values[val->op_ids[1]];
            a->grad += b->data * val->grad;
            b->grad += a->data * val->grad;
        }
        break;

        case Operation::DIV:
        {
            auto a = &_engine.values[val->op_ids[0]];
            auto b = &_engine.values[val->op_ids[1]];
            a->grad += (1.0 / b->data) * val->grad;
            b->grad += (a->data / sqr(b->data)) * val->grad;
        }
        break;

        case Operation::POW:
        {
            auto a = &_engine.values[val->op_ids[0]];
            auto b = &_engine.values[val->op_ids[1]];
            a->grad += (b->data * pow(a->data, b->data - 1.0)) * val->grad;
            b->grad += pow(val->grad, b->data) * log(a->data) * val->grad;
        }
        break;

        case Operation::RELU:
        {
            auto a = &_engine.values[val->op_ids[0]];
            a->grad += (a->data > 0.0) * val->grad;
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
    const Value& true_val = _engine.values[val.id];
    printf("Value(data=%f, grad=%f, op=%s)\n", true_val.data, true_val.grad, OP_LABELS[(int)true_val.op]);
}

void val_backward(Value val)
{
    size_t* queue = (size_t*) malloc(_engine.num_values * sizeof(size_t));
    bool* visited = (bool*) calloc(_engine.num_values, 1);

    _engine.values[val.id].grad = 1.0;

    size_t queue_tail = 1;
    size_t queue_head = 0;
    queue[0] = val.id;
    while(queue_head != queue_tail)
    {
        Value* cval = &_engine.values[queue[queue_head]];
        queue_head = (queue_head + 1) % _engine.num_values;
        if(!visited[cval->id])
        {
            _backward(cval);
            //val_print(*cval);

            visited[cval->id] = true;
            for(size_t ci = 0; ci < 2; ++ci)
            {
                if(cval->op_ids[ci] != size_t(-1))
                {
                    queue[queue_tail] = cval->op_ids[ci];
                    queue_tail = (queue_tail + 1) % _engine.num_values;
                }
            }
        }
    }

    free(visited);
    free(queue);
}

void engine_init(void)
{
    _engine.num_values  = 0;
    _engine.size_values = 10;
    _engine.values = (Value*) malloc(_engine.size_values * sizeof(Value));
}

void engine_free(void)
{
    free(_engine.values);
}
