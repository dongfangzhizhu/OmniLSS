"""Deep GAMLSS implementation

使用神经网络建模分布参数，这是 OmniLSS 超越 R gamlss 的创新功能。

Deep GAMLSS 的核心思想：
- 传统 GAMLSS: μ = g^(-1)(β₀ + β₁x + s(z))
- Deep GAMLSS: μ = g^(-1)(NN_μ(x, z))

优势：
1. 自动学习复杂的非线性关系
2. 不需要手动指定平滑项
3. 可以处理高维数据
4. 自动学习交互效应

Examples
--------
>>> import jax.numpy as jnp
>>> from omnilss.deep import fit_deep_gamlss
>>> from omnilss import NO
>>>
>>> # 生成数据
>>> X = jnp.array([[1, 2], [3, 4], [5, 6]])
>>> y = jnp.array([1.5, 3.2, 5.1])
>>>
>>> # 拟合 Deep GAMLSS
>>> model, params = fit_deep_gamlss(X, y, family=NO())
>>>
>>> # 预测
>>> pred_params = model.apply(params, X)
>>> print(pred_params["mu"])
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax


class ParameterNetwork(nn.Module):
    """参数网络

    为单个分布参数建模的神经网络。

    Parameters
    ----------
    hidden_dims : tuple of int
        隐藏层维度，例如 (64, 32) 表示两个隐藏层
    activation : Callable
        激活函数，默认为 ReLU
    use_batch_norm : bool
        是否使用 Batch Normalization
    dropout_rate : float
        Dropout 比率，0 表示不使用 dropout

    Examples
    --------
    >>> net = ParameterNetwork(hidden_dims=(64, 32))
    >>> params = net.init(jax.random.PRNGKey(0), jnp.ones((10, 5)))
    >>> output = net.apply(params, jnp.ones((10, 5)))
    >>> print(output.shape)  # (10,)
    """

    hidden_dims: Sequence[int] = (64, 32)
    activation: Callable = nn.relu
    use_batch_norm: bool = False
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, training: bool = False):
        """前向传播

        Parameters
        ----------
        x : jnp.ndarray
            输入特征 (batch_size, n_features)
        training : bool
            是否在训练模式

        Returns
        -------
        output : jnp.ndarray
            输出 (batch_size,)
        """
        for i, dim in enumerate(self.hidden_dims):
            x = nn.Dense(dim, name=f"dense_{i}")(x)

            if self.use_batch_norm:
                x = nn.BatchNorm(use_running_average=not training, name=f"bn_{i}")(x)

            x = self.activation(x)

            if self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        # 输出层
        x = nn.Dense(1, name="output")(x)
        return x.squeeze(-1)


class DeepGAMLSS(nn.Module):
    """Deep GAMLSS model

    使用神经网络建模所有分布参数。

    Parameters
    ----------
    family : FamilyDefinition
        分布族
    hidden_dims : tuple of int
        隐藏层维度
    activation : Callable
        激活函数
    use_batch_norm : bool
        是否使用 Batch Normalization
    dropout_rate : float
        Dropout 比率
    shared_layers : bool
        是否在参数之间共享底层

    Examples
    --------
    >>> from omnilss import NO
    >>> model = DeepGAMLSS(family=NO(), hidden_dims=(64, 32))
    >>> params = model.init(jax.random.PRNGKey(0), jnp.ones((10, 5)))
    >>> pred_params = model.apply(params, jnp.ones((10, 5)))
    >>> print(pred_params.keys())  # dict_keys(['mu', 'sigma'])
    """

    family: Any
    hidden_dims: Sequence[int] = (64, 32)
    activation: Callable = nn.relu
    use_batch_norm: bool = False
    dropout_rate: float = 0.0
    shared_layers: bool = False

    def setup(self):
        """设置网络结构"""
        # 获取可估计的参数
        self.param_names = self.family.estimable_parameters

        if self.shared_layers:
            # 共享底层：为全部 hidden_dims 创建独立的 Dense 层列表
            # 不使用 ParameterNetwork，以避免其自带的输出层与 param_heads 重复
            self.shared_dense_layers = [
                nn.Dense(dim, name=f"shared_dense_{i}")
                for i, dim in enumerate(self.hidden_dims)
            ]
            # 每个参数有独立的输出头（仅 Dense(1)，无激活）
            self.param_heads = {
                param: nn.Dense(1, name=f"head_{param}") for param in self.param_names
            }
        else:
            # 每个参数有独立的网络
            self.param_networks = {
                param: ParameterNetwork(
                    hidden_dims=self.hidden_dims,
                    activation=self.activation,
                    use_batch_norm=self.use_batch_norm,
                    dropout_rate=self.dropout_rate,
                )
                for param in self.param_names
            }

    def __call__(self, x, training: bool = False):
        """前向传播

        Parameters
        ----------
        x : jnp.ndarray
            输入特征 (batch_size, n_features)
        training : bool
            是否在训练模式

        Returns
        -------
        params : dict
            预测的分布参数 {param_name: values}
        """
        params = {}

        if self.shared_layers:
            # 通过共享隐藏层逐层前向传播
            h = x
            for dense_layer in self.shared_dense_layers:
                h = dense_layer(h)  # 线性变换
                h = self.activation(h)  # 非线性激活
                # 按需应用 Dropout（仅在训练阶段生效）
                if self.dropout_rate > 0:
                    h = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(
                        h
                    )
            # 每个参数用独立输出头，再经逆链接函数映射到参数空间
            for param in self.param_names:
                eta = self.param_heads[param](h).squeeze(-1)
                params[param] = self.family.link_inverses[param](eta)
        else:
            # 独立网络
            for param in self.param_names:
                eta = self.param_networks[param](x, training=training)
                # 应用链接函数的逆
                params[param] = self.family.link_inverses[param](eta)

        return params


def fit_deep_gamlss(
    X: jnp.ndarray,
    y: jnp.ndarray,
    family: Any,
    hidden_dims: Sequence[int] = (64, 32),
    activation: Callable = nn.relu,
    use_batch_norm: bool = False,
    dropout_rate: float = 0.0,
    shared_layers: bool = False,
    learning_rate: float = 0.001,
    n_epochs: int = 100,
    batch_size: Optional[int] = None,
    validation_split: float = 0.0,
    early_stopping_patience: int = 10,
    verbose: bool = True,
    random_seed: int = 0,
) -> Tuple[DeepGAMLSS, Any, Dict[str, list]]:
    """拟合 Deep GAMLSS 模型

    Parameters
    ----------
    X : jnp.ndarray
        输入特征 (n_samples, n_features)
    y : jnp.ndarray
        响应变量 (n_samples,)
    family : FamilyDefinition
        分布族
    hidden_dims : tuple of int
        隐藏层维度
    activation : Callable
        激活函数
    use_batch_norm : bool
        是否使用 Batch Normalization
    dropout_rate : float
        Dropout 比率
    shared_layers : bool
        是否在参数之间共享底层
    learning_rate : float
        学习率
    n_epochs : int
        训练轮数
    batch_size : int, optional
        批大小，None 表示使用全批次
    validation_split : float
        验证集比例 (0-1)
    early_stopping_patience : int
        早停的耐心值
    verbose : bool
        是否打印训练进度
    random_seed : int
        随机种子

    Returns
    -------
    model : DeepGAMLSS
        训练好的模型
    params : dict
        模型参数
    history : dict
        训练历史 (loss, val_loss)

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from omnilss.deep import fit_deep_gamlss
    >>> from omnilss import NO
    >>>
    >>> # 生成数据
    >>> np.random.seed(42)
    >>> n = 1000
    >>> X = np.random.randn(n, 2)
    >>> y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + np.random.randn(n) * 0.3
    >>>
    >>> # 拟合模型
    >>> model, params, history = fit_deep_gamlss(
    ...     jnp.array(X), jnp.array(y), family=NO(),
    ...     hidden_dims=(64, 32),
    ...     learning_rate=0.001,
    ...     n_epochs=100,
    ...     verbose=True
    ... )
    """
    # 转换为 JAX 数组
    X = jnp.array(X)
    y = jnp.array(y)

    n_samples = X.shape[0]

    # 划分训练集和验证集
    if validation_split > 0:
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val

        # 随机打乱
        key = jax.random.PRNGKey(random_seed)
        perm = jax.random.permutation(key, n_samples)
        X = X[perm]
        y = y[perm]

        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]
    else:
        X_train, y_train = X, y
        X_val, y_val = None, None

    # 创建模型
    model = DeepGAMLSS(
        family=family,
        hidden_dims=hidden_dims,
        activation=activation,
        use_batch_norm=use_batch_norm,
        dropout_rate=dropout_rate,
        shared_layers=shared_layers,
    )

    # 初始化参数
    key = jax.random.PRNGKey(random_seed)
    params = model.init(key, X_train[:1], training=False)

    # 创建优化器
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    # 定义损失函数
    def loss_fn(params, X, y, training=False):
        """负对数似然损失"""
        pred_params = model.apply(params, X, training=training)  # type: ignore[arg-type]
        # 计算负对数似然
        log_lik = family.d(x=y, log=True, **pred_params)  # type: ignore[arg-type]
        return -jnp.mean(log_lik)

    # JIT 编译训练步骤
    @jax.jit
    def train_step(params, opt_state, X_batch, y_batch):
        """单步训练"""
        loss, grads = jax.value_and_grad(loss_fn)(
            params, X_batch, y_batch, training=True
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # 训练循环
    history = {"loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0
    val_loss = float("inf")  # 初始化防止 possibly unbound（无验证集时不会更新）

    if verbose:
        print(f"\n{'=' * 70}")
        print("训练 Deep GAMLSS")
        print(f"{'=' * 70}")
        print(f"模型: {family.name}")
        print(
            f"样本数: {n_samples} (训练: {len(X_train)}, 验证: {len(X_val) if X_val is not None else 0})"
        )
        print(f"特征数: {X.shape[1]}")
        print(f"隐藏层: {hidden_dims}")
        print(f"学习率: {learning_rate}")
        print(f"批大小: {batch_size if batch_size else '全批次'}")
        print(f"{'=' * 70}\n")

    for epoch in range(n_epochs):
        # 训练
        if batch_size is None:
            # 全批次
            params, opt_state, train_loss = train_step(
                params, opt_state, X_train, y_train
            )
        else:
            # 小批次
            n_batches = int(np.ceil(len(X_train) / batch_size))
            epoch_losses = []

            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(X_train))
                X_batch = X_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]

                params, opt_state, batch_loss = train_step(
                    params, opt_state, X_batch, y_batch
                )
                epoch_losses.append(batch_loss)

            train_loss = jnp.mean(jnp.array(epoch_losses))

        history["loss"].append(float(train_loss))

        # 验证
        if X_val is not None:
            val_loss = loss_fn(params, X_val, y_val, training=False)
            history["val_loss"].append(float(val_loss))

            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"\n早停于 epoch {epoch + 1}")
                    break

        # 打印进度
        if verbose and (epoch + 1) % 10 == 0:
            if X_val is not None:
                print(
                    f"Epoch {epoch + 1:4d}/{n_epochs}: "
                    f"Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}"
                )
            else:
                print(f"Epoch {epoch + 1:4d}/{n_epochs}: Loss = {train_loss:.4f}")

    if verbose:
        print(f"\n{'=' * 70}")
        print("训练完成")
        print(f"{'=' * 70}\n")

    return model, params, history


def predict_deep_gamlss(model: DeepGAMLSS, params: Dict, X: jnp.ndarray) -> Any:
    """使用 Deep GAMLSS 模型进行预测

    Parameters
    ----------
    model : DeepGAMLSS
        训练好的模型
    params : dict
        模型参数
    X : jnp.ndarray
        输入特征 (n_samples, n_features)

    Returns
    -------
    pred_params : dict
        预测的分布参数 {param_name: values}

    Examples
    --------
    >>> pred_params = predict_deep_gamlss(model, params, X_new)
    >>> print(pred_params["mu"])
    """
    X = jnp.array(X)
    pred_params = model.apply(params, X, training=False)
    return pred_params
