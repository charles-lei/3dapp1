package com.recon3d.repository

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

@Dao
interface TaskDao {
    @Query("SELECT * FROM tasks ORDER BY createdAt DESC")
    fun observeAll(): Flow<List<TaskEntity>>

    @Query("SELECT * FROM tasks WHERE taskId = :taskId")
    fun observeById(taskId: String): Flow<TaskEntity?>

    @Query("SELECT * FROM tasks WHERE taskId = :taskId")
    suspend fun getById(taskId: String): TaskEntity?

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsert(entity: TaskEntity)

    @Query("DELETE FROM tasks WHERE taskId = :taskId")
    suspend fun delete(taskId: String)
}
