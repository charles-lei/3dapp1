package com.recon3d.capture

import java.io.File

class SessionPhotoStore(
    private val rootDir: File,
    private val clock: () -> Long = System::currentTimeMillis,
) {
    fun newSession(): File = File(rootDir, clock().toString()).apply { mkdirs() }

    fun photos(sessionDir: File): List<File> =
        sessionDir.listFiles { f -> f.extension.lowercase() == "jpg" }?.sortedBy { it.name } ?: emptyList()

    fun count(sessionDir: File): Int = photos(sessionDir).size

    fun delete(sessionDir: File) {
        sessionDir.deleteRecursively()
    }
}
