#include "nn.h"
//�����AShift-JIS�œǂݍ���ł���Ȃ�AUTF-8�œǂݍ���ł�������
//上のコメントは、UTF-8で読み込んでいるなら無視してください。
#define epoch 10      // エポック回数
#define batsz 100     // バッチサイズ
#define learning 0.1 // 学習率
float err1 = 0.0, err2 = 0.0;
void print(int m, int n, const float *mat)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf(" %.4f ", mat[i * n + j]);
        }
        printf("\n");
    }
} // �?ストデータ、訓練�?ータそれぞれの交差エントロピ�?�誤差の総和
void fc(int m, int n, const float *x, const float *A, const float *b, float *y) // m*n行�?�とnベクトルからmベクトル生�??
{
    for (int i = 0; i < m; i++)
    {
        float yk = 0.0;
        for (int j = 0; j < n; j++)
        {
            yk += A[i * n + j] * x[j];
        }
        y[i] = yk + b[i];
    }
}
void relu(int n, const float *x, float *y) // y[i] = max(0,x[i]) をや�?
{
    for (int i = 0; i < n; i++)
    {
        if (x[i] > 0)
            y[i] = x[i];
        else
            y[i] = 0.0;
    }
}
void softmax(int n, const float *x, float *y) // nベクトル実数値から、nベクトル[0,1]の実数へと変換
{
    float sum = 0, mx = 0; // exp(x[i])の合計とx[i]の最大値
    for (int i = 0; i < n; i++)
    {
        if (x[i] > mx)
            mx = x[i]; // x[i]の最大値の導�?�
    }
    for (int i = 0; i < n; i++)
    {
        sum += exp(x[i] - mx); // overflow 防止のため、mxを引く
    }
    for (int i = 0; i < n; i++)
    {
        y[i] = exp(x[i] - mx) / (sum); // overflow防止のため、mxを引く
    }
}
void softmaxwithloss_bwd(int n, const float *y, unsigned char t, float *dEdx) // softmax層の�?伝播
{
    // t はOne - Hot ベクトル表現なのに注意す�?
    for (int k = 0; k < n; k++)
    {
        if (k == t)
        {
            dEdx[k] = y[k] - 1.0;
        }
        else
        {
            dEdx[k] = y[k];
        }
    }
}
void relu_bwd(int n, const float *x, const float *dEdy, float *dEdx) // relu層の�?伝播
{
    for (int i = 0; i < n; i++)
    {
        if (x[i] > 0)
            dEdx[i] = dEdy[i];
        else
            dEdx[i] = 0.0;
    }
}
void fc_bwd(int m, int n, const float *x, const float *dEdy, const float *A, float *dEdA, float *dEdb, float *dEdx) // fc層の�?伝播
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            dEdA[i * n + j] = dEdy[i] * x[j];
        }
        dEdb[i] = dEdy[i];
    }
    for (int i = 0; i < n; i++)
    { // m*n行�?�でnベクトルに�?伝播することに注意す�?
        dEdx[i] = 0;
        for (int j = 0; j < m; j++)
        {
            dEdx[i] += A[i + j * n] * dEdy[j];
        }
    }
}
// 12 添え字をランダ�?にシャ�?フルすることで�?配�?�をシャ�?フルしたことにする�?
void shuffle(int n, int *x)
{
    for (int i = 0; i < n; i++)
    {
        int med = x[i];
        int rad = rand() % n; // iとrandomな値をswapする
        x[i] = x[rad];
        x[rad] = med;
    }
}
// 14
void add(int n, const float *x, float *o) // 配�?�oに配�?�xの値を�?��?��?�添え字に対応する形で�?算す�?
{
    for (int i = 0; i < n; i++)
    {
        o[i] += x[i];
    }
}
void scale(int n, float x, float *o) // 配�?�oに一定値xをかける
{
    for (int i = 0; i < n; i++)
    {
        o[i] *= x;
    }
}
void init(int n, float x, float *o) // 配�?�oを一定値xで初期�?
{
    for (int i = 0; i < n; i++)
    {
        o[i] = x;
    }
}
void rand_init(int n, float *o) // 配�?�oを[-1,1]の値で初期�?
{
    for (int i = 0; i < n; i++)
    {
        o[i] = ((float)rand() / RAND_MAX) * 2 - 1.0;
    }
    // o[i] �? [-1:1] の乱数で初期�?
}
float generate_gaussian(float mean, float variance) // ガウス�?�?の乱数
{
    static int alternate = 0;
    static float stored_value;
    if (alternate == 0)
    {
        float u1, u2, v1, v2, s;
        do
        {
            u1 = 2.0 * rand() / RAND_MAX - 1.0; //-1から1の実数乱数に変換
            u2 = 2.0 * rand() / RAND_MAX - 1.0;
            s = u1 * u1 + u2 * u2; // 二乗和
        } while (s >= 1.0 || s == 0.0);
        v1 = u1 * sqrt(-2.0 * log(s) / s);
        v2 = u2 * sqrt(-2.0 * log(s) / s);
        stored_value = v2 * sqrt(variance) + mean;
        alternate = 1;

        return v1 * sqrt(variance) + mean;
    }
    else
    {
        alternate = 0;
        return stored_value;
    }
}
void He_init(int n, int a, float *o) // Heの初期値 平�?0�?散sqrt(2/a)の正規�??�?の乱数で初期�?
{
    float var = (2 / (float)a);
    for (int i = 0; i < n; i++)
    {
        o[i] = generate_gaussian(0, var);
    } // Heの初期値
}
void add_at_once(const int n, const int sec, const int thi, const int m, const float *x1, const float *x2, const float *x3, const float *x4, const float *x5, const float *x6, float *o1, float *o2, float *o3, float *o4, float *o5, float *o6)
{ // 6層ともなると、コードが冗長になる�?�で一つにまとめる�?
    add(n * sec, x1, o1);
    add(sec, x2, o2);
    add(sec * thi, x3, o3);
    add(thi, x4, o4);
    add(thi * m, x5, o5);
    add(m, x6, o6);
}
// 13 交差エントロピ�?�を�?��?
float cross_entropy_error(const float *y, int t) // tはOne-hotベクトルより�?1である部�?�?けを抜き出せ�?�よい�?
{
    return -log(y[t] + 1e-7); // Undefined Error防止のため微小量を足�?
}
// 18　�?ータのセーブとロー�?
void save(const char *filename, int m, int n, const float *A, const float *b)
{
    float *out = malloc(m * (n + 1) * sizeof(float)); // Aとbを一つの配�?�に結合する
    FILE *file = fopen(filename, "wb+");
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            out[i * n + j] = A[i * n + j];
        }
        out[n * m + i] = b[i];
    }
    fwrite(out, sizeof(float), m * (n + 1), file);
    fclose(file);
}
void load(const char *filename, int m, int n, float *A, float *b)
{
    FILE *file = fopen(filename, "r");
    float *input = (float *)malloc((n + 1) * m * sizeof(float));
    int sz = fread(input, sizeof(float), m * (n + 1), file);
    if (sz == 0)
    {
        printf("NULL\n");
    }
    fclose(file);
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            A[i * n + j] = input[i * n + j];
        }
        b[i] = input[n * m + i];
    }
} // 読み込んだbmpファイルから推論した結果を表示(�?数対�?)
void inference(const float *A1, const float *b1, const float *A2, const float *b2, const float *A3, const float *b3, const float *ts_x, int test_count, const int sec, const int thi)
{
    float *x1 = malloc(784 * sizeof(float));
    float *y1 = malloc(sec * sizeof(float));
    float *y2 = malloc(thi * sizeof(float));
    float *y3 = malloc(10 * sizeof(float));
    for (int i = 0; i < test_count; i++)
    {
        for (int j = 0; j < 784; j++)
        {
            x1[j] = ts_x[784 * i + j];
        }
        fc(sec, 784, x1, A1, b1, y1);
        relu(sec, y1, y1);

        fc(thi, sec, y1, A2, b2, y2);
        relu(thi, y2, y2);

        fc(10, thi, y2, A3, b3, y3);
        softmax(10, y3, y1);

        float mx = 0.0;
        int ans = 0;
        for (int j = 0; j < 10; j++)
        {
            if (y1[j] > mx)
            {
                mx = y1[j];
                ans = j;
            }
        }
        printf("%d\n", ans);
    }
    free(x1);
    free(y1);
    free(y2);
    free(y3);
}
// 16推論動�?
void inference6(const float *A1, const float *b1, const float *A2, const float *b2, const float *A3, const float *b3, const float *tr_x, const unsigned char *tr_y, const int train_count, const float *ts_x, unsigned char *ts_y, int test_count, const int sec, const int thi)
{
    float sum = 0.0, sum2 = 0.0;//test とtrainの正解数
    float *x1 = malloc(784 * sizeof(float));//�?伝播時�?�途中計算保持用
    float *y1 = malloc(sec * sizeof(float));
    float *y2 = malloc(thi * sizeof(float));
    float *y3 = malloc(10 * sizeof(float));
    for (int i = 0; i < test_count; i++)
    {
        for (int j = 0; j < 784; j++)
        {
            x1[j] = ts_x[784 * i + j];
        }
        fc(sec, 784, x1, A1, b1, y1);
        relu(sec, y1, y1);
        fc(thi, sec, y1, A2, b2, y2);
        relu(thi, y2, y2);
        fc(10, thi, y2, A3, b3, y3);
        softmax(10, y3, y1);
        float mx = 0.0;
        int ans = 0;
        for (int j = 0; j < 10; j++)
        {
            if (y1[j] > mx)
            {
                mx = y1[j];
                ans = j;
            }
        }
        if (ans == ts_y[i])
            sum += 1.0;
        err1 += cross_entropy_error(y1, ans);
    }
    for (int i = 0; i < train_count; i++) // 過学習して�?な�?か、trainの場合�?�正解�?もチェ�?クする
    {
        for (int j = 0; j < 784; j++)
        {
            x1[j] = tr_x[784 * i + j];
        }
        fc(sec, 784, x1, A1, b1, y1);
        relu(sec, y1, y1);
        fc(thi, sec, y1, A2, b2, y2);
        relu(thi, y2, y2);
        fc(10, thi, y2, A3, b3, y3);
        softmax(10, y3, y1);
        float mx = 0.0;
        int ans = 0;
        for (int j = 0; j < 10; j++)
        {
            if (y1[j] > mx)
            {
                mx = y1[j];
                ans = j;
            }
        }
        if (ans == tr_y[i])
            sum2 += 1.0;
        err2 += cross_entropy_error(y1, ans); // 交差エントロピ�?�誤差の合計を計算す�?
    }
    printf("test_acc %.5f train_acc %.5f\ntest_error %.5f train_error %.5f\n", (float)sum * 100 / test_count, (float)sum2 * 100 / train_count, err1, err2);
    err1 = 0.0, err2 = 0.0;
    free(x1);
    free(y1);
    free(y2);
    free(y3);
}
// 誤差�?伝播
void backward6(const float *A1, const float *b1, const float *A2, const float *b2, const float *A3, const float *b3, const float *x, unsigned char t, float *y, float *dEdA1, float *dEdb1, float *dEdA2, float *dEdb2, float *dEdA3, float *dEdb3, const int sec, const int thi)
{
    // dEdxはfc層から次に誤差を伝播させるため�?�値,dEdyはreluからdEdxへとつなげるためのバッファー。おそらく、�?要な�?が、安�?�性のため入れて�?�?
    int n = 784, m = 10; // Y1�?け大�?字なのはy1にするとなぜか警告がでたため。途中計算保持用
    float Y1[sec], y2[thi], y3[m], dEdx3[m], dEdy3[thi], dEdx2[sec], dEdy2[sec], dEdx1[n], dEdy1[n];
    fc(sec, n, x, A1, b1, Y1);
    relu(sec, Y1, y);
    fc(thi, sec, y, A2, b2, y2);
    relu(thi, y2, y);
    fc(m, thi, y, A3, b3, y3);
    softmax(m, y3, y); // ここまで�?伝播

    // ここから�?伝播
    softmaxwithloss_bwd(m, y, t, dEdx3);
    fc_bwd(m, thi, y2, dEdx3, A3, dEdA3, dEdb3, dEdy3);
    relu_bwd(thi, y2, dEdy3, dEdx2);
    fc_bwd(thi, sec, Y1, dEdx2, A2, dEdA2, dEdb2, dEdy2);
    relu_bwd(sec, Y1, dEdy2, dEdx1);
    fc_bwd(sec, n, x, dEdx1, A1, dEdA1, dEdb1, dEdy1);
}
// 17
void SGD6(const int init_check, float *A1, float *b1, float *A2, float *b2, float *A3, float *b3, const int m, const int n, const int train_count, const float *tr_x, const unsigned char *tr_y, float *ts_x, unsigned char *ts_y, const int test_count, const int second, const int third)
{
    // train�?ータ一つ一つの誤差
    float dEdA1[second * n], dEdb1[second], dEdA2[second * third], dEdb2[third], dEdA3[m * third], dEdb3[m];
    // バッチサイズの誤差の平�?計算保持用
    float AvedEdA1[second * n], AvedEdb1[second], AvedEdA2[second * third], AvedEdb2[third], AvedEdA3[m * third], AvedEdb3[m];
    float *X = malloc(batsz * n * sizeof(float));
    float y[m];
    int index[train_count]; // Xはバッチ�??り取り用。yは純伝播計算用 また、indexはSGDでランダ�?に配�?�にアクセスするためのも�?�
    srand(0);               // 乱数初期�?
    if (init_check == 0)    // 初期化したいとき�?�最初に0を�?��?
    {
        He_init(second * n, n, A1); // A1 ~ b3を�?�期�?(Heの初期値) 15.4に該�?
        He_init(second, n, b1);
        He_init(second * third, second, A2);
        He_init(third, second, b2);
        He_init(m * third, third, A3);
        He_init(m, third, b3);
    }
    srand(0);
    for (int ep = 0; ep < epoch; ep++)
    {
        for (int i = 0; i < train_count; i++)
            index[i] = i;
        //  15.5.(a)
        shuffle(train_count, index);
        for (int i = 0; i < train_count / batsz; i++)
        { // 15.5.b.(i)
            init(n * second, 0.0, AvedEdA1);
            init(second, 0.0, AvedEdb1);
            init(second * third, 0.0, AvedEdA2);
            init(third, 0.0, AvedEdb2);
            init(third * m, 0.0, AvedEdA3);
            init(m, 0.0, AvedEdb3);
            for (int j = 0; j < batsz; j++)
            { // 15.5.(b).ii
                for (int k = 0; k < n; k++)
                    X[k] = tr_x[index[i * batsz + j] * n + k]; // シャ�?フルした添え字�?�入�?(画�?)を保持
                // 15.5.(b).iii �?伝播&平�?勾配に�?�?
                backward6(A1, b1, A2, b2, A3, b3, X, tr_y[index[i * batsz + j]], y, dEdA1, dEdb1, dEdA2, dEdb2, dEdA3, dEdb3, second, third);
                add_at_once(n, second, third, m, dEdA1, dEdb1, dEdA2, dEdb2, dEdA3, dEdb3, AvedEdA1, AvedEdb1, AvedEdA2, AvedEdb2, AvedEdA3, AvedEdb3);
            } // 15.5.(b).iv
            scale(second, -1.0 * learning / (float)batsz, AvedEdb1);
            scale(n * second, -1.0 * learning / (float)batsz, AvedEdA1);
            scale(third, -1.0 * learning / (float)batsz, AvedEdb2);
            scale(second * third, -1.0 * learning / (float)batsz, AvedEdA2);
            scale(m, -1.0 * learning / (float)batsz, AvedEdb3);
            scale(m * third, -1.0 * learning / (float)batsz, AvedEdA3);
            // 15.5.(b).vb
            add_at_once(n, second, third, m, AvedEdA1, AvedEdb1, AvedEdA2, AvedEdb2, AvedEdA3, AvedEdb3, A1, b1, A2, b2, A3, b3);
        }
        printf("Step%d ;", ep);
        inference6(A1, b1, A2, b2, A3, b3, tr_x, tr_y, train_count, ts_x, ts_y, test_count, second, third);
    }
    free(X);
}
int main(int argc, char *argv[])
{
    float *train_x = NULL;
    unsigned char *train_y = NULL;
    int train_count = -1;

    float *test_x = NULL;
    unsigned char *test_y = NULL;
    int test_count = -1;

    int width = -1;
    int height = -1;

    load_mnist(&train_x, &train_y, &train_count,
               &test_x, &test_y, &test_count,
               &width, &height);

/* 浮動小数点例外で停止することを確認するため�?�コー�? */
#if 0
  volatile float x = 0;
  volatile float y = 0;
  volatile float z = x/y;
#endif

    int second = 50, third = 100; // �?れ層の大きさ
    // fc層のための配�??
    float *A1 = malloc(784 * second * sizeof(float));
    float *b1 = malloc(second * sizeof(float));
    float *A2 = malloc(second * third * sizeof(float));
    float *b2 = malloc(third * sizeof(float));
    float *A3 = malloc(third * 10 * sizeof(float));
    float *b3 = malloc(10 * sizeof(float));

    if (argc == 1)
    {
        float *x = load_mnist_bmp(argv[0]);
        load(argv[1], second, 784, A1, b1);
        load(argv[2], third, second, A2, b2);
        load(argv[3], 10, third, A3, b3);
        inference(A1, b1, A2, b2, A3, b3, x, 1, second, third);
    }
    else if (argc == 0)
    {
        SGD6(0, A1, b1, A2, b2, A3, b3, 10, 784, train_count, train_x, train_y, test_x, test_y, test_count, second, third);
        save(argv[0], second, 784, A1, b1);
        save(argv[1], third, second, A2, b2);
        save(argv[2], 10, third, A3, b3);
    }
    free(A1);
    free(b1);
    free(A2);
    free(b2);
    free(A3);
    free(b3);
    return 0;
}
