import tensorflow as tf
import numpy as np
import time
import os

# ===================== 1. 加载数据集 =====================
print("=== 加载MNIST手写数字数据集 ===")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)
print(f"训练集数量: {len(x_train)}, 测试集数量: {len(x_test)}")

# ===================== 2. 搭建轻量化模型 =====================
print("\n=== 搭建轻量化CNN模型 ===")
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(8, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# ===================== 3. 训练模型 =====================
print("\n=== 开始训练模型 ===")
history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# 测试原始模型准确率
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n原始浮点模型测试准确率: {test_acc:.4f}")

# 保存原始模型并查看体积
model.save("original_model.h5")
original_size = os.path.getsize("original_model.h5") / 1024
print(f"原始浮点模型体积: {original_size:.2f} KB")

# ===================== 4. TinyML核心：INT8动态范围量化 =====================
print("\n=== 执行INT8权重量化，生成TinyML模型 ===")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
# 开启默认优化，自动将权重压缩为INT8
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 转换并保存量化模型
tflite_model = converter.convert()
with open("tinyml_model_quantized.tflite", "wb") as f:
    f.write(tflite_model)

quantized_size = os.path.getsize("tinyml_model_quantized.tflite") / 1024
print(f"量化后TinyML模型体积: {quantized_size:.2f} KB")
print(f"体积压缩倍率: {original_size / quantized_size:.1f} 倍")

# ===================== 5. 量化模型精度与速度验证 =====================
print("\n=== 验证量化模型性能 ===")

interpreter = tf.lite.Interpreter(model_path="tinyml_model_quantized.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

correct = 0
test_num = 1000
start_time = time.time()

for i in range(test_num):
    # 浮点输入，和原始模型预处理一致，无需额外转换
    input_data = np.expand_dims(x_test[i], axis=0).astype(np.float32)
    
    # 执行推理
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # 获取预测结果
    output_data = interpreter.get_tensor(output_details[0]['index'])
    pred = np.argmax(output_data)
    
    if pred == y_test[i]:
        correct += 1

total_time = time.time() - start_time
quantized_acc = correct / test_num
avg_infer_time = total_time / test_num * 1000

# ===================== 6. 最终结果对比 =====================
print("\n" + "="*50)
print("TinyML项目最终性能对比报告")
print("="*50)
print(f"原始浮点模型 | 准确率: {test_acc:.4f} | 体积: {original_size:.2f} KB")
print(f"TinyML量化模型 | 准确率: {quantized_acc:.4f} | 体积: {quantized_size:.2f} KB")
print(f"单样本平均推理时间: {avg_infer_time:.3f} ms")
print(f"准确率损失: {(test_acc - quantized_acc)*100:.2f} 个百分点")
print(f"体积压缩倍率: {original_size / quantized_size:.1f} 倍")
print("="*50)
print("\n项目运行完成！生成的tinyml_model_quantized.tflite即为TinyML轻量化模型文件。")