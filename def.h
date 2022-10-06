// #define DEBUG_MODE 1

#define ACTIVATION &vrelu
#define ACTIVATION_D &vrelu_d
#define OUT_ACTIVATION &softmax
#define OUT_ACTIVATION_D 0
#define pad(n, k, out) pad_zero(n, k, out)

#define PARAM_MIN -1.0
#define PARAM_MAX 1.0

#define LEARN_RATE 1