package com.recon3d.capture

import java.io.File

class SessionPhotoStore(
    private val rootDir: File,
    private val clock: () -> Long = System::currentTimeMillis,
) {
    fun newSession(): File = File(rootDir, clock().toString()).apply { mkdirs() }
        .also { check(it.isDirectory) { "无法创建采集目录: ${it.absolutePath}" } }

    /**
     * 返回 sessionDir 下按名称排序的 .jpg 文件。
     * 若 sessionDir 不存在或无效，返回空列表（调用方按零张照片处理）。
     */
    fun photos(sessionDir: File): List<File> =
        sessionDir.listFiles { f -> f.extension.lowercase() == "jpg" }?.sortedBy { it.name } ?: emptyList()

    fun count(sessionDir: File): Int = photos(sessionDir).size

    fun delete(sessionDir: File) {
        sessionDir.deleteRecursively()
    }
}
