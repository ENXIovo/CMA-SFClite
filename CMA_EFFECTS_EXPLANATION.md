# 因果中介分析（CMA）的三种效应

## 📚 理论基础

在因果中介分析中，我们关注三种效应：

### 1. Total Effect (TE) - 总效应
```
TE = Y(X_source) - Y(X_base)
```
**含义**：只改变输入，不干预任何中间层，测量输出的总变化。

**示例**：
```python
p(she | "The man said that") - p(she | "The nurse said that")
```

### 2. Natural Indirect Effect (NIE) - 自然间接效应
```
NIE = Y(X_base, M_source) - Y(X_base, M_base)
```
**含义**：保持输入为 base，只将中间变量（注意力头）替换为 source 的值。

**示例**：
```python
p(she | "The nurse said that", head_from_man) - p(she | "The nurse said that", head_from_nurse)
```

### 3. Natural Direct Effect (NDE) - 自然直接效应
```
NDE = Y(X_source, M_base) - Y(X_base, M_base)
```
**含义**：改变输入，但保持中间变量不变（冻结在 base 的值）。

**示例**：
```python
p(she | "The man said that", head_from_nurse) - p(she | "The nurse said that", head_from_nurse)
```

---

## 🔗 关系

```
TE = NIE + NDE
```

即：总效应 = 间接效应（通过中间变量） + 直接效应（绕过中间变量）

---

## 🎯 当前代码实现

### `cma_gender_bias.py` 计算的是 **NIE**

```python
# 步骤 1：在 source 上保存注意力头的值
with model.trace(source_prompt):  # "The man said that"
    z_src = model.transformer.h[layer_idx].attn.c_proj.input[...].save()

# 步骤 2：在 base 上获取未干预的概率
with model.trace(base_prompt):  # "The nurse said that"
    logits_clean = model.output.logits[...].save()
    
# 步骤 3：在 base 上干预（替换为 source 的头）
with model.trace(base_prompt):  # 仍然是 "The nurse said that"
    model.transformer.h[layer_idx].attn.c_proj.input[...] = z_src
    logits_intervened = model.output.logits[...].save()

# NIE = p(intervened) - p(clean)
NIE = p_intervened - p_clean
```

这是**正确的 NIE 计算方法**！

---

## 💡 解释

### NIE 为正（红色）的含义
```
NIE > 0  =>  将 source 的头移植到 base 后，增加了 p(she)
```

**说明**：该注意力头在 source 输入（"man"）上学到了某种表征，当移植到 base 输入（"nurse"）时，反而增加了"she"的概率。这表明：
- 该头并非单纯编码输入词本身
- 而是编码了某种**与输入词相关的性别信息**
- 当这个头从"man"转移到"nurse"时，它携带的信息导致模型更倾向于预测"she"

这正是**性别偏见的证据**：该头将职业词（nurse）与性别（female）刻板关联。

---

## 🔍 为什么 NIE 比 TE 更有用？

### TE（总效应）
```
TE = p(she | "The man said that") - p(she | "The nurse said that")
```
- 只告诉我们：输入改变导致输出改变了多少
- 但**无法定位**是哪些中间层/注意力头负责

### NIE（间接效应）
```
NIE = p(she | "The nurse", head_from_man) - p(she | "The nurse", head_from_nurse)
```
- 告诉我们：**每个注意力头**对偏见的贡献
- 可以**精确定位**负责偏见的头（如 Head 0.6, 10.5）
- 可以通过**消融这些头**来减少偏见

---

## 📊 论文中的用法

Vig et al. 2020 使用的正是 **NIE**，因为：
1. 可以绘制 层×头 的热力图
2. 定位具体的偏见编码位置
3. 为后续干预（如头消融）提供指导

---

## 🚀 如何计算 TE？

如果需要 TE 作为参考基线：

```python
# 在 main() 中已经计算了！
with model.trace(base_prompt):
    logits_base = model.output.logits[0, -1, :].save()

with model.trace(source_prompt):
    logits_src = model.output.logits[0, -1, :].save()

# TE（总效应）
TE = logits_src.softmax(dim=-1)[pronoun_id].item() - logits_base.softmax(dim=-1)[pronoun_id].item()
```

当前代码在"[步骤 1] 检查基础偏见"部分已经计算并展示了 TE！

---

## ✅ 总结

| 效应 | 当前代码 | 用途 |
|-----|---------|------|
| **TE** | ✅ 已计算（步骤 1） | 确认存在偏见 |
| **NIE** | ✅ 主要分析 | 定位偏见来源（热力图） |
| **NDE** | ❌ 未计算 | 理论完整性（TE = NIE + NDE） |

**结论**：当前实现是完全正确的！NIE 是 CMA 的核心，用于精确定位哪些注意力头编码了性别偏见。

