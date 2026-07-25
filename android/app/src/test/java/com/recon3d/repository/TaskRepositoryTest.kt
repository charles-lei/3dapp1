package com.recon3d.repository

import android.content.Context
import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import com.recon3d.network.ErrorBody
import com.recon3d.network.TaskStatusDto
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.test.runTest
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class TaskRepositoryTest {
    private lateinit var db: AppDatabase
    private lateinit var repo: TaskRepository

    @Before fun setup() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        db = Room.inMemoryDatabaseBuilder(context, AppDatabase::class.java)
            .allowMainThreadQueries()
            .build()
        repo = TaskRepository(db.taskDao())
    }

    @After fun tearDown() = db.close()

    private fun status(id: String, state: String, progress: Int = 0, error: ErrorBody? = null) =
        TaskStatusDto(id, state, progress, "detail", null, error)

    @Test fun `insert and observe all ordered by createdAt desc`() = runTest {
        repo.insert(TaskEntity("a", createdAt = 1, state = "uploaded"))
        repo.insert(TaskEntity("b", createdAt = 2, state = "done"))
        val tasks = repo.all.first()
        assertEquals(listOf("b", "a"), tasks.map { it.taskId })
    }

    @Test fun `upsertStatus preserves local fields`() = runTest {
        repo.insert(TaskEntity("t1", createdAt = 5, state = "uploaded", photoDir = "/p", glbPath = null))
        repo.upsertStatus(status("t1", "training", progress = 55))
        val t = repo.byId("t1").first()!!
        assertEquals("training", t.state)
        assertEquals(55, t.progress)
        assertEquals("/p", t.photoDir)
        assertEquals(5, t.createdAt)
    }

    @Test fun `upsertStatus stores error fields`() = runTest {
        repo.insert(TaskEntity("t2", 1, "uploaded"))
        repo.upsertStatus(status("t2", "failed", error = ErrorBody("TRAIN_FAILED", "训练失败")))
        val t = repo.byId("t2").first()!!
        assertEquals("TRAIN_FAILED", t.errorCode)
        assertEquals("训练失败", t.errorMessageZh)
    }

    @Test fun `markGlb sets path`() = runTest {
        repo.insert(TaskEntity("t3", 1, "done"))
        repo.markGlb("t3", "/models/t3.glb")
        assertEquals("/models/t3.glb", repo.byId("t3").first()!!.glbPath)
    }

    @Test fun `delete removes entity`() = runTest {
        repo.insert(TaskEntity("t4", 1, "done"))
        repo.delete("t4")
        assertNull(repo.byId("t4").first())
    }
}
