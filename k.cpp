#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>
#include <chrono>
#include <random>
using namespace std::chrono;

// consts
const size_t kAlphSize = 256;
const size_t kBlockSize = 16;
const size_t kKeysCount = 10;
const size_t kCodeSize = 500 * 1024 * 1024;
const uint8_t kPoly = 0xCE;

template<typename T>
using Matrix = std::vector<std::vector<T>>;

const std::vector<uint8_t> kNonLinConv = {
    0xFC, 0xEE, 0xDD, 0x11, 0xCF, 0x6E, 0x31, 0x16, 0xFB, 0xC4, 0xFA, 0xDA, 0x23, 0xC5, 0x04, 0x4D,
    0xE9, 0x77, 0xF0, 0xDB, 0x93, 0x2E, 0x99, 0xBA, 0x17, 0x36, 0xF1, 0xBB, 0x14, 0xCD, 0x5F, 0xC1,
    0xF9, 0x18, 0x65, 0x5A, 0xE2, 0x5C, 0xEF, 0x21, 0x81, 0x1C, 0x3C, 0x42, 0x8B, 0x01, 0x8E, 0x4F,
    0x05, 0x84, 0x02, 0xAE, 0xE3, 0x6A, 0x8F, 0xA0, 0x06, 0x0B, 0xED, 0x98, 0x7F, 0xD4, 0xD3, 0x1F,
    0xEB, 0x34, 0x2C, 0x51, 0xEA, 0xC8, 0x48, 0xAB, 0xF2, 0x2A, 0x68, 0xA2, 0xFD, 0x3A, 0xCE, 0xCC,
    0xB5, 0x70, 0x0E, 0x56, 0x08, 0x0C, 0x76, 0x12, 0xBF, 0x72, 0x13, 0x47, 0x9C, 0xB7, 0x5D, 0x87,
    0x15, 0xA1, 0x96, 0x29, 0x10, 0x7B, 0x9A, 0xC7, 0xF3, 0x91, 0x78, 0x6F, 0x9D, 0x9E, 0xB2, 0xB1,
    0x32, 0x75, 0x19, 0x3D, 0xFF, 0x35, 0x8A, 0x7E, 0x6D, 0x54, 0xC6, 0x80, 0xC3, 0xBD, 0x0D, 0x57,
    0xDF, 0xF5, 0x24, 0xA9, 0x3E, 0xA8, 0x43, 0xC9, 0xD7, 0x79, 0xD6, 0xF6, 0x7C, 0x22, 0xB9, 0x03,
    0xE0, 0x0F, 0xEC, 0xDE, 0x7A, 0x94, 0xB0, 0xBC, 0xDC, 0xE8, 0x28, 0x50, 0x4E, 0x33, 0x0A, 0x4A,
    0xA7, 0x97, 0x60, 0x73, 0x1E, 0x00, 0x62, 0x44, 0x1A, 0xB8, 0x38, 0x82, 0x64, 0x9F, 0x26, 0x41,
    0xAD, 0x45, 0x46, 0x92, 0x27, 0x5E, 0x55, 0x2F, 0x8C, 0xA3, 0xA5, 0x7D, 0x69, 0xD5, 0x95, 0x3B,
    0x07, 0x58, 0xB3, 0x40, 0x86, 0xAC, 0x1D, 0xF7, 0x30, 0x37, 0x6B, 0xE4, 0x88, 0xD9, 0xE7, 0x89,
    0xE1, 0x1B, 0x83, 0x49, 0x4C, 0x3F, 0xF8, 0xFE, 0x8D, 0x53, 0xAA, 0x90, 0xCA, 0xD8, 0x85, 0x61,
    0x20, 0x71, 0x67, 0xA4, 0x2D, 0x2B, 0x09, 0x5B, 0xCB, 0x9B, 0x25, 0xD0, 0xBE, 0xE5, 0x6C, 0x52,
    0x59, 0xA6, 0x74, 0xD2, 0xE6, 0xF4, 0xB4, 0xC0, 0xD1, 0x66, 0xAF, 0xC2, 0x39, 0x4B, 0x63, 0xB6    
};


const std::vector<uint8_t> kNonLinConvRev = {
    0xA5, 0x2D, 0x32, 0x8F, 0x0E, 0x30, 0x38, 0xC0, 0x54, 0xE6, 0x9E, 0x39, 0x55, 0x7E, 0x52, 0x91,
    0x64, 0x03, 0x57, 0x5A, 0x1C, 0x60, 0x07, 0x18, 0x21, 0x72, 0xA8, 0xD1, 0x29, 0xC6, 0xA4, 0x3F,
    0xE0, 0x27, 0x8D, 0x0C, 0x82, 0xEA, 0xAE, 0xB4, 0x9A, 0x63, 0x49, 0xE5, 0x42, 0xE4, 0x15, 0xB7,
    0xC8, 0x06, 0x70, 0x9D, 0x41, 0x75, 0x19, 0xC9, 0xAA, 0xFC, 0x4D, 0xBF, 0x2A, 0x73, 0x84, 0xD5,
    0xC3, 0xAF, 0x2B, 0x86, 0xA7, 0xB1, 0xB2, 0x5B, 0x46, 0xD3, 0x9F, 0xFD, 0xD4, 0x0F, 0x9C, 0x2F,
    0x9B, 0x43, 0xEF, 0xD9, 0x79, 0xB6, 0x53, 0x7F, 0xC1, 0xF0, 0x23, 0xE7, 0x25, 0x5E, 0xB5, 0x1E,
    0xA2, 0xDF, 0xA6, 0xFE, 0xAC, 0x22, 0xF9, 0xE2, 0x4A, 0xBC, 0x35, 0xCA, 0xEE, 0x78, 0x05, 0x6B,
    0x51, 0xE1, 0x59, 0xA3, 0xF2, 0x71, 0x56, 0x11, 0x6A, 0x89, 0x94, 0x65, 0x8C, 0xBB, 0x77, 0x3C,
    0x7B, 0x28, 0xAB, 0xD2, 0x31, 0xDE, 0xC4, 0x5F, 0xCC, 0xCF, 0x76, 0x2C, 0xB8, 0xD8, 0x2E, 0x36,
    0xDB, 0x69, 0xB3, 0x14, 0x95, 0xBE, 0x62, 0xA1, 0x3B, 0x16, 0x66, 0xE9, 0x5C, 0x6C, 0x6D, 0xAD,
    0x37, 0x61, 0x4B, 0xB9, 0xE3, 0xBA, 0xF1, 0xA0, 0x85, 0x83, 0xDA, 0x47, 0xC5, 0xB0, 0x33, 0xFA,
    0x96, 0x6F, 0x6E, 0xC2, 0xF6, 0x50, 0xFF, 0x5D, 0xA9, 0x8E, 0x17, 0x1B, 0x97, 0x7D, 0xEC, 0x58,
    0xF7, 0x1F, 0xFB, 0x7C, 0x09, 0x0D, 0x7A, 0x67, 0x45, 0x87, 0xDC, 0xE8, 0x4F, 0x1D, 0x4E, 0x04,
    0xEB, 0xF8, 0xF3, 0x3E, 0x3D, 0xBD, 0x8A, 0x88, 0xDD, 0xCD, 0x0B, 0x13, 0x98, 0x02, 0x93, 0x80,
    0x90, 0xD0, 0x24, 0x34, 0xCB, 0xED, 0xF4, 0xCE, 0x99, 0x10, 0x44, 0x40, 0x92, 0x3A, 0x01, 0x26,
    0x12, 0x1A, 0x48, 0x68, 0xF5, 0x81, 0x8B, 0xC7, 0xD6, 0x20, 0x0A, 0x08, 0x00, 0x4C, 0xD7, 0x74
};

const Matrix<uint8_t> kLinConv = {
    { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
    { 1, 148, 32, 133, 16, 194, 192, 1, 251, 1, 192, 194, 16, 133, 32, 148 },
};

union Block {
    uint8_t bytes[kBlockSize];
    __uint128_t full;
};

std::istream &operator>>(std::istream &is, Block &dt) {
    for (size_t i = 0; i < kBlockSize; ++i) {
        is >> dt.bytes[i];
    }
    return is;
}

constexpr uint8_t Multiply(uint8_t a, uint8_t b) {
    uint8_t ans = 0;
    while (b != 0) {
        if ((b & 1) == 1) {
            ans ^= a;
        }
        uint8_t aMaxBit = (a & 0x80);
        a <<= 1;
        if (aMaxBit == 0x80) {
            a ^= kPoly;
        }
        b >>= 1;
    }
    return ans;
}

inline Matrix<uint8_t> GenerateFieldMultTable() {
    Matrix<uint8_t> table(kAlphSize, std::vector<uint8_t>(kAlphSize));
    for (uint8_t i = 1; i != 0; ++i) {
        for (uint8_t j = 1; j < i; ++j) {
            uint8_t mul = Multiply(i, j);
            table[i][j] = mul;
            table[j][i] = mul;
        }
        table[i][i] = Multiply(i, i);
    }
    return table;
}

inline Matrix<uint8_t> MatrixSquare(const Matrix<uint8_t> &matrix, const Matrix<uint8_t> &multTable) {
    Matrix<uint8_t> ans(matrix.size(), std::vector<uint8_t>(matrix.size()));
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix.size(); ++j) {
            for (size_t k = 0; k < matrix.size(); ++k) {
                ans[i][j] ^= multTable[matrix[i][k]][matrix[k][j]];
            }
        }
    }
    return ans;
}

inline Matrix<uint8_t> GetLinearConversion(const Matrix<uint8_t> &multTable) {
    auto currentConv = kLinConv;
    for (size_t i = 0; i < 4; ++i) {
        currentConv = MatrixSquare(currentConv, multTable);
    }
    return currentConv;
}

inline Matrix<Block> GetLSMatrix(const Matrix<uint8_t> &multTable) {
    Matrix<Block> LS(kBlockSize, std::vector<Block>(kAlphSize));
    
    auto linearConv = GetLinearConversion(multTable);
    for (size_t i = 0; i < kBlockSize; ++i) {
        for (size_t j = 0; j < kAlphSize; ++j) {
            for (size_t k = 0; k < kBlockSize; ++k) {
                LS[i][j].bytes[k] = multTable[linearConv[k][i]][kNonLinConv[j]];
            }
        }
    }
    return LS;
}

constexpr void NonLin(Block *arg) {
    for (size_t i = 0; i < kBlockSize; ++i) {
        arg->bytes[i] = kNonLinConv[arg->bytes[i]];
    }
}

constexpr void Shift(Block *arg, const Matrix<uint8_t> &multTable) {
    Block helper = *arg;
    arg->bytes[kBlockSize - 1] = 0;
    for (size_t i = 0; i < kBlockSize; ++i) {
        if (i != 0) {
            arg->bytes[i - 1] = helper.bytes[i];
        }
        arg->bytes[kBlockSize - 1] ^= multTable[helper.bytes[i]][kLinConv[15][i]];
    }
}

constexpr void Lin(Block *arg, const Matrix<uint8_t> &multTable) {
    for (size_t i = 0; i < 16; ++i) {
        Shift(arg, multTable);
    }
}

inline void LSCalc(Block *arg, const Matrix<Block> &LS) {
    Block hint = *arg;
    arg->full = 0;
    for (size_t i = 0; i < kBlockSize; ++i) {
        arg->full ^= LS[i][hint.bytes[i]].full;
    }
}

inline std::vector<Block> GenerateConstants(const Matrix<uint8_t> &multTable) {
    std::vector<Block> constants(2 * kBlockSize);
    for (size_t i = 0; i < 2 * kBlockSize; ++i) {
        constants[i].bytes[0] = i + 1;
        Lin(&constants[i], multTable);
    }
    return constants;
}

inline std::vector<uint8_t> GetRandomKey() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint8_t> distrib;
    std::vector<uint8_t> ans;
    for (size_t i = 0; i < 2 * kBlockSize; ++i) {
        ans.push_back(distrib(gen));
    }
    return ans;
}

inline std::pair<Block, Block> Feystel(const Block &left, const Block &right, 
const Block &constant, const Matrix<Block> &LS) {
    Block ansRight = left;
    Block ansLeft = left;
    ansLeft.full ^= constant.full;
    LSCalc(&ansLeft, LS);
    ansLeft.full ^= right.full;
    return {ansLeft, ansRight};
}

inline std::vector<Block> GenerateKeys(const std::vector<uint8_t> &key, 
const Matrix<uint8_t> &multTable, const Matrix<Block> &LS) {
    std::vector<Block> keys(kKeysCount);

    auto constants = GenerateConstants(multTable);
    for (size_t i = 0; i < key.size(); ++i) {
        keys[i / 16].bytes[i % 16] = key[i];
    }
    auto helper = std::make_pair(keys[0], keys[1]);
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 8; ++j) {
            helper = Feystel(helper.first, helper.second, constants[8 * i + j], LS);
        }
        keys[2 * i + 2] = helper.first;
        keys[2 * i + 3] = helper.second;
    }
    return keys;
}

inline void LSCalc(Block *arg, __uint128_t *LS, __uint128_t *k) {
    arg->full = LS[arg->bytes[0]] ^ LS[0x0100 ^ arg->bytes[1]] ^ 
    LS[0x200 ^ arg->bytes[2]] ^ LS[0x300 ^ arg->bytes[3]] ^ LS[0x400 ^ arg->bytes[4]] ^ 
    LS[0x500 ^ arg->bytes[5]] ^ LS[0x600 ^ arg->bytes[6]] ^ LS[0x700 ^ arg->bytes[7]] ^ 
    LS[0x800 ^ arg->bytes[8]] ^ LS[0x900 ^ arg->bytes[9]] ^ LS[0xa00 ^ arg->bytes[10]] ^ 
    LS[0xb00 ^ arg->bytes[11]] ^ LS[0xc00 ^ arg->bytes[12]] ^ LS[0xd00 ^ arg->bytes[13]] ^
    LS[0xe00 ^ arg->bytes[14]] ^ LS[0xf00 ^ arg->bytes[15]] ^ *k;
}

void Enc(Block *begin, Block *end, const std::vector<Block> &keys, __uint128_t *LS) {
    __uint128_t key = keys[0].full;
    for (Block *bl = begin; bl != end; ++bl) {
        bl->full ^= key;
    }
    for (size_t i = 1; i < 10; ++i) {
        key = keys[i].full;
        for (Block *bl = begin; bl != end; ++bl) {
            LSCalc(bl, LS, &key);
        }
    }
}

uint8_t *GenerateRandomCode() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint8_t> distrib;
    uint8_t *code = new uint8_t[kCodeSize];
    for (size_t i = 0; i < kCodeSize; ++i) {
        code[i] = distrib(gen);
    }
    return code;
}

inline void MakeOneDimLS(__uint128_t *oneDimLS, const Matrix<Block> &LS) {
    for (size_t i = 0; i < 16; ++i) {
        for (size_t j = 0; j < 16; ++j) {
            oneDimLS[(i << 8) ^ j] = LS[i][j].full;
        }
    }    
}

inline void StatEncrypt(const std::vector<Block> &keys, __uint128_t *LS) {
    auto code = GenerateRandomCode();
    std::cout << "Start Encoding: " << kCodeSize << " byte (500 MB)" << std::endl;
    auto start = high_resolution_clock::now();
    Enc(reinterpret_cast<Block*>(code), reinterpret_cast<Block *>(code + kCodeSize), keys, LS);
    auto microsecs = duration_cast<microseconds>(high_resolution_clock::now() - start).count();
    std::cout <<  microsecs << " microseconds(" << microsecs / 1e6 << " seconds)" << std::endl;
    delete[] code;
}

int main() {
    auto multTable = GenerateFieldMultTable();
    auto LS = GetLSMatrix(multTable);
    auto keys = GenerateKeys(GetRandomKey(), multTable, LS);
    __uint128_t *oneDimLS = new __uint128_t[16 * 256];
    MakeOneDimLS(oneDimLS, LS);

    StatEncrypt(keys, oneDimLS);
    
    delete[] oneDimLS;
}