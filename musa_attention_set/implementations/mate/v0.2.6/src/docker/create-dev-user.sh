#!/usr/bin/env bash
set -euo pipefail

username="${1:-}"
user_uid="${2:-}"
user_gid="${3:-}"

if [[ ! "${username}" =~ ^[a-z_][a-z0-9_-]*[$]?$ ]]; then
    echo "Invalid development username: ${username}" >&2
    exit 1
fi
for value_name in user_uid user_gid; do
    value="${!value_name}"
    if [[ ! "${value}" =~ ^[0-9]+$ ]] || [[ "${value}" == "0" ]]; then
        echo "${value_name} must be a non-zero numeric ID" >&2
        exit 1
    fi
done

if getent group "${user_gid}" >/dev/null; then
    user_group="$(getent group "${user_gid}" | cut -d: -f1)"
elif getent group "${username}" >/dev/null; then
    existing_gid="$(getent group "${username}" | cut -d: -f3)"
    echo "Group ${username} already exists with GID ${existing_gid}" >&2
    exit 1
else
    user_group="${username}"
    groupadd --gid "${user_gid}" "${user_group}"
fi

if getent passwd "${user_uid}" >/dev/null; then
    existing_user="$(getent passwd "${user_uid}" | cut -d: -f1)"
    if [[ "${existing_user}" != "${username}" ]]; then
        echo "UID ${user_uid} is already assigned to ${existing_user}" >&2
        exit 1
    fi
    usermod --gid "${user_group}" --shell /bin/zsh "${username}"
elif id "${username}" >/dev/null 2>&1; then
    existing_uid="$(id -u "${username}")"
    echo "User ${username} already exists with UID ${existing_uid}" >&2
    exit 1
else
    useradd --uid "${user_uid}" --gid "${user_group}" --create-home \
        --shell /bin/zsh "${username}"
fi

printf '%s ALL=(root) NOPASSWD:ALL\n' "${username}" \
    > "/etc/sudoers.d/${username}"
chmod 0440 "/etc/sudoers.d/${username}"
visudo --check --file "/etc/sudoers.d/${username}" >/dev/null

if getent group video >/dev/null; then
    usermod --append --groups video "${username}"
fi
if getent group render >/dev/null; then
    render_group=render
elif getent group 109 >/dev/null; then
    render_group="$(getent group 109 | cut -d: -f1)"
else
    echo "Neither the render group nor GID 109 exists in the base image" >&2
    exit 1
fi
usermod --append --groups "${render_group}" "${username}"

user_home="/home/${username}"
install -d -m 0755 -o "${username}" -g "${user_group}" \
    /workspace/mate \
    "${user_home}/.cache" \
    "${user_home}/.local" \
    "${user_home}/.local/bin" \
    "${user_home}/.local/share"
chown "${username}:${user_group}" /workspace /workspace/mate "${user_home}"

ln -s /opt/mate/miniforge3 "${user_home}/miniforge3"
ln -s /opt/mate/nvm "${user_home}/.nvm"
ln -s /opt/mate/oh-my-zsh "${user_home}/.oh-my-zsh"

# The variables below are intentionally expanded when zsh reads the file.
# shellcheck disable=SC2016
printf '%s\n' \
    'export ZSH="/opt/mate/oh-my-zsh"' \
    'ZSH_THEME="ys"' \
    'plugins=(git zsh-autosuggestions)' \
    'source "$ZSH/oh-my-zsh.sh"' \
    'export PATH="$HOME/.venv/bin:$HOME/.local/bin:/opt/mate/bin:/opt/mate/miniforge3/envs/mate/bin:/opt/mate/miniforge3/bin:$PATH"' \
    'export NVM_DIR="/opt/mate/nvm"' \
    '[ -s "$NVM_DIR/nvm.sh" ] && source "$NVM_DIR/nvm.sh"' \
    'export NPM_CONFIG_PREFIX="$HOME/.local"' \
    > "${user_home}/.zshrc"
chown "${username}:${user_group}" "${user_home}/.zshrc"
chown -h "${username}:${user_group}" \
    "${user_home}/miniforge3" \
    "${user_home}/.nvm" \
    "${user_home}/.oh-my-zsh"

runuser --user "${username}" -- \
    env HOME="${user_home}" \
    "${CONDA_ENV}/bin/python" -m venv --system-site-packages \
    "${user_home}/.venv"
