-- Run after 003_create_rls_policies.sql.
-- Storage RLS policies for buckets created in scripts/setup_storage.py.
-- The FastAPI backend uses the service_role key which bypasses RLS; these
-- policies protect anon-key access from Flutter (supabase_flutter).

-- chat-attachments: public read, authenticated owner-scoped write.
-- Writes must land under {auth.uid()}/... so users can only upload into
-- their own folder. No DELETE policy yet — retention is a follow-up.

DROP POLICY IF EXISTS "chat_attachments_public_read" ON storage.objects;
CREATE POLICY "chat_attachments_public_read" ON storage.objects
    FOR SELECT
    USING (bucket_id = 'chat-attachments');

DROP POLICY IF EXISTS "chat_attachments_owner_write" ON storage.objects;
CREATE POLICY "chat_attachments_owner_write" ON storage.objects
    FOR INSERT TO authenticated
    WITH CHECK (
        bucket_id = 'chat-attachments'
        AND auth.uid()::text = (storage.foldername(name))[1]
    );
