# Tilelang Musa Develop Guide

## 构建方式
```Shell
# 进入仓库主目录
cd $tilelang_musa

# 构建 Debug 版本
pip install -e . -v --no-build-isolation -C cmake.build-type=Debug

# 构建 Release 版本
pip install -e . -v --no-build-isolation

# 构建 whl 包
python -m build
```

## commit 约定
* 架构相关的公共 commit message 使用 `musa:` 前缀。
* 针对某个架构 `mp_*` 的 commit message 使用 `mp_*:` 前缀。
* 架构无关的 commit message 使用 `tilelang:` 前缀。
* commit message 统一使用小写字母。

## 测试约定
* 公共测试用例放在 `testing/musa/common`，使用 `@tilelang.testing.requires_musa` 约束。
* 架构相关测试用例放在 `testing/musa/mp_*`, 使用 `@tilelang.testing.requires_musa_compute_version_le` 约束。

## 开发约定
* 架构特性原则上只能调用 mtcc 提供的 MUSA C 接口，不能依赖 mutlass/mute；如果 mtcc 未提供对应的 MUSA C 接口，可以调用 builtin 并加上注释说明。
* 实现与 TileLang CUDA 后端相同或相近的 MUSA 功能时，应优先参考并尽量保持 CUDA 的用户语义、IR 表达、lowering 结构和测试方式；硬件接口或能力存在差异时，保留 MUSA 专属实现并明确说明差异，避免机械照搬 CUDA 代码。
* `__MUSA_ARCH__` 用于设备端代码判断当前编译架构，例如 device template、`__device__` 函数、设备指令选择和 warp size 判断；`__MUSA_ARCH_LIST__` 用于编译入口判断本次编译目标，例如是否 include 架构头文件或声明模板入口，不用于设备端执行分支。
* device template 按 `src/tl_templates/musa/common/`、`src/tl_templates/musa/mp_*` 组织；公共能力放在 `common`，架构专属能力放在对应的 `mp_*`。
* MUSA backend 的适配应保持后端隔离，原则上只修改 MUSA 专属代码和测试，尽量不要修改Tilelang原生逻辑。
