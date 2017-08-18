# coding: utf-8
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import scipy.misc as misc
import os
import time
from collections import namedtuple
from skimage.measure import compare_mse, compare_nrmse, compare_ssim, compare_psnr
from util.ops import conv2d, deconv2d, lrelu, fc, batch_norm
from util.dataset import TrainDataProvider, InjectDataProvider
from util.utils import scale_back, merge, save_concat_images, save_image


LossHandle = namedtuple("LossHandle",["d_loss", "g_loss", "const_loss", "l1_loss", "cheat_loss",
                                      "tv_loss"])
InputHandle = namedtuple("InputHandle", ["real_data"])
EvalHandle = namedtuple("EvalHandle", ["encoder", "generator", "target", "source"])
SummaryHandle = namedtuple("SummaryHandle", ["d_merged", "g_merged"])


class Font2Font(object):
    def __init__(self, experiment_dir=None, experiment_id=0, batch_size=16, input_width=256, output_width=256,
                 generator_dim=64, discriminator_dim=64, L1_penalty=100, Lconst_penalty=15, Ltv_penalty=0.0,
                 input_filters=1, output_filters=1):
        self.experiment_dir = experiment_dir
        self.experiment_id = experiment_id
        self.batch_size = batch_size
        self.input_width = input_width
        self.output_width = output_width
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim
        self.L1_penalty = L1_penalty
        self.Lconst_penalty = Lconst_penalty
        self.Ltv_penalty = Ltv_penalty
        self.input_filters = input_filters
        self.output_filters = output_filters
        # init all the directories
        self.sess = None
        # experiment_dir is needed for training
        if experiment_dir:
            self.data_dir = os.path.join(self.experiment_dir, "data")
            self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoint")
            self.sample_dir = os.path.join(self.experiment_dir, "sample")
            self.log_dir = os.path.join(self.experiment_dir, "logs")
            # if not exist, make those path first
            if not os.path.exists(self.checkpoint_dir):
                os.mkdir(self.checkpoint_dir)
            if not os.path.exists(self.sample_dir):
                os.mkdir(self.sample_dir)
            if not os.path.exists(self.log_dir):
                os.mkdir(self.log_dir)

    def encoder(self, images, is_training, reuse=False):
        with tf.variable_scope("generator"):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            encode_layers = dict()

            def encode_layer(x, output_filters, layer):
                act = lrelu(x)
                conv = conv2d(act, output_filters=output_filters, scope="g_e%d_conv" % layer)
                enc = batch_norm(conv, is_training, scope="g_e%d_bn" % layer)
                encode_layers["e%d" % layer] = enc
                return enc

            e1 = conv2d(images, self.generator_dim, scope="g_e1_conv")
            encode_layers["e1"] = e1

            e2 = encode_layer(e1, self.generator_dim * 2, 2)
            e3 = encode_layer(e2, self.generator_dim * 4, 3)
            e4 = encode_layer(e3, self.generator_dim * 8, 4)
            e5 = encode_layer(e4, self.generator_dim * 8, 5)
            e6 = encode_layer(e5, self.generator_dim * 8, 6)
            e7 = encode_layer(e6, self.generator_dim * 8, 7)
            e8 = encode_layer(e7, self.generator_dim * 8, 8)

            return e8, encode_layers

    def decoder(self, encoded, encoding_layers, is_training, reuse=False):
        with tf.variable_scope("generator"):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            s = self.output_width
            s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

            def decode_layer(x, output_width, output_fileters, layer, enc_layer, dropout=False, do_concat=True):
                dec = deconv2d(tf.nn.relu(x), [self.batch_size, output_width, output_width, output_fileters],
                               scope="g_d%d_deconv" % layer)
                if layer != 8:
                    # IMPORTANT: normalization for last layer
                    # Very important, otherwise GAN is unstable
                    dec = batch_norm(dec, is_training, scope="g_d%d_bn" % layer)
                if dropout:
                    dec = tf.nn.dropout(dec, 0.5)
                if do_concat:
                    dec = tf.concat([dec, enc_layer], 3)
                return dec
            d1 = decode_layer(encoded, s128, self.generator_dim * 8, layer=1, enc_layer=encoding_layers["e7"],
                              dropout=True)
            d2 = decode_layer(d1, s64, self.generator_dim * 8, layer=2, enc_layer=encoding_layers["e6"], dropout=True)
            d3 = decode_layer(d2, s32, self.generator_dim * 8, layer=3, enc_layer=encoding_layers["e5"], dropout=True)
            d4 = decode_layer(d3, s16, self.generator_dim * 8, layer=4, enc_layer=encoding_layers["e4"])
            d5 = decode_layer(d4, s8, self.generator_dim * 4, layer=5, enc_layer=encoding_layers["e3"])
            d6 = decode_layer(d5, s4, self.generator_dim * 2, layer=6, enc_layer=encoding_layers["e2"])
            d7 = decode_layer(d6, s2, self.generator_dim, layer=7, enc_layer=encoding_layers["e1"])
            d8 = decode_layer(d7, s, self.output_filters, layer=8, enc_layer=None, do_concat=False)

            output = tf.nn.tanh(d8) # scale to (-1, 1)
            return output

    def generator(self, images, is_training, reuse=False):
        e8, enc_layers = self.encoder(images, is_training=is_training, reuse=reuse)
        output = self.decoder(e8, enc_layers, is_training=is_training, reuse=reuse)
        return output, e8

    def discriminator(self, image, is_training, reuse=False):
        with tf.variable_scope("discriminator"):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            # [batch,256,256,1] -> [batch,128,128,64]
            h0 = lrelu(conv2d(image, self.discriminator_dim, scope="d_h0_conv"))
            # [batch,128,128,64] -> [batch,64,64,64*2]
            h1 = lrelu(batch_norm(conv2d(h0, self.discriminator_dim * 2, scope="d_h1_conv"),
                                  is_training, scope="d_bn_1"))
            # [batch,64,64,64*2] -> [batch,32,32,64*4]
            h2 = lrelu(batch_norm(conv2d(h1, self.discriminator_dim * 4, scope="d_h2_conv"),
                                  is_training, scope="d_bn_2"))
            # [batch,32,32,64*4] -> [batch,31,31,64*8]
            h3 = lrelu(batch_norm(conv2d(h2, self.discriminator_dim * 8, sh=1, sw=1, scope="d_h3_conv"),
                                  is_training, scope="d_bn_3"))
            # real or fake binary loss
            fc1 = fc(tf.reshape(h3, [self.batch_size, -1]), 1, scope="d_fc1")

            return tf.sigmoid(fc1), fc1

    def build_model(self, is_training=True):
        real_data = tf.placeholder(tf.float32, [self.batch_size, self.input_width, self.input_width,
                                                self.input_filters + self.output_filters], name="real_A_and_B_images")
        # target images
        real_B = real_data[:, :, :, :self.input_filters]
        # source images
        real_A = real_data[:, :, :, self.input_filters:self.input_filters + self.output_filters]

        # fake images
        fake_B, encoded_real_A = self.generator(real_A, is_training=is_training)
        real_B_generated, _ = self.generator(real_B, is_training=is_training, reuse=True) # keep the invariance of G

        real_AB = tf.concat([real_A, real_B], 3)
        fake_AB = tf.concat([real_A, fake_B], 3)
        real_AB_generated = tf.concat([real_A, real_B_generated], 3)

        # Note it is not possible to set reuse flag back to False
        # initialize all variables before setting resue to True
        real_D, real_D_logits = self.discriminator(real_AB, is_training=is_training, reuse=False)
        fake_D, fake_D_logits = self.discriminator(fake_AB, is_training=is_training, reuse=True)
        real_D_generated, real_D_logits_generated = self.discriminator(real_AB_generated, is_training=is_training,
                                                                       reuse=True)

        # encoding constant loss
        # this loss assume that generated imaged and real image
        # should reside in the same space and close to each other
        encoded_fake_B = self.encoder(fake_B, is_training, reuse=True)[0]
        const_loss = (tf.reduce_mean(tf.square(encoded_real_A - encoded_fake_B))) * self.Lconst_penalty

        # L1 loss between real and generated imags
        l1_loss = self.L1_penalty * tf.reduce_mean(tf.abs(fake_B - real_B))

        # total variation loss
        width = self.output_width
        tv_loss = (tf.nn.l2_loss(fake_B[:, 1:, :, :] - fake_B[:, :width - 1, :, :]) / width
                   + tf.nn.l2_loss(fake_B[:, :, 1:, :] - fake_B[:, :, :width - 1, :]) /width) * self.Ltv_penalty

        # binary real/fake loss
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_D_logits,
                                                                             labels=tf.ones_like(real_D)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_D_logits,
                                                                             labels=tf.zeros_like(fake_D)))
        d_loss_real_generated = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_D_logits_generated,
                                                                                       labels=tf.ones_like(real_D_generated)))
        # maximize the chance generator to fool the discriminator
        cheat_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_D_logits,
                                                                            labels=tf.ones_like(fake_D)))

        # synthesize all losses to d_loss and g_loss
        d_loss = d_loss_real + d_loss_fake + d_loss_real_generated

        g_loss = l1_loss + const_loss + tv_loss + cheat_loss

        # summary all losses
        l1_loss_summary = tf.summary.scalar("l1_loss", l1_loss)
        const_loss_summary = tf.summary.scalar("const_loss_summary", const_loss)
        cheat_loss_summary = tf.summary.scalar("cheat_loss_summary", cheat_loss)
        tv_loss_summary = tf.summary.scalar("tv_loss_summary", tv_loss)

        d_loss_summary = tf.summary.scalar("d_loss", d_loss)
        g_loss_summary = tf.summary.scalar("g_loss", g_loss)

        d_merged_summary = tf.summary.merge([d_loss_summary])
        g_merged_summary = tf.summary.merge([l1_loss_summary, const_loss_summary, cheat_loss_summary, tv_loss_summary,
                                             g_loss_summary])

        # expose useful node in the graph as handles globally
        input_handle = InputHandle(real_data=real_data)
        loss_handle = LossHandle(d_loss=d_loss, g_loss=g_loss, const_loss=const_loss, cheat_loss=cheat_loss,
                                 l1_loss=l1_loss, tv_loss=tv_loss)
        eval_handle = EvalHandle(encoder=encoded_real_A, generator=fake_B, target=real_B, source=real_A)
        summary_handle = SummaryHandle(d_merged=d_merged_summary, g_merged=g_merged_summary)

        # those operation will be shared, so we need to make them visible globally
        setattr(self, "input_handle", input_handle)
        setattr(self, "loss_handle", loss_handle)
        setattr(self, "eval_handle", eval_handle)
        setattr(self, "summary_handle", summary_handle)

    def register_session(self, sess):
        self.sess = sess

    def retrieve_trainable_vars(self, freeze_encoder=False):
        t_vars = tf.trainable_variables()

        d_vars = [var for var in t_vars if "d_" in var.name]
        g_vars = [var for var in t_vars if "g_" in var.name]

        if freeze_encoder:
            # exclude encoder weights
            print("freeze encoder weights")
            g_vars = [var for var in g_vars if not ("g_e" in var.name)]
        return g_vars, d_vars

    def retrieve_generator_vars(self):
        all_vars = tf.global_variables()
        generator_vars = [var for var in all_vars if "embedding" in var.name or "g_" in var.name]
        return generator_vars

    def retrieve_handle(self):
        input_handle = getattr(self, "input_handle")
        loss_handle = getattr(self, "loss_handle")
        eval_handle = getattr(self, "eval_handle")
        summary_handle = getattr(self, "summary_handle")
        return input_handle, loss_handle, eval_handle, summary_handle

    def get_model_id_and_dir(self):
        model_id = "experiment_%d_batch_%d" % (self.experiment_id, self.batch_size)
        model_dir = os.path.join(self.checkpoint_dir, model_id)
        return model_id, model_dir

    def checkpoint(self, saver, step):
        model_name = "font2font.model"
        model_id, model_dir = self.get_model_id_and_dir()

        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        saver.save(self.sess, os.path.join(model_dir, model_name), global_step=step)

    def restore_model(self, saver, model_dir):
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("restore model %s" % model_dir)
        else:
            print("fail to restore model %s" % model_dir)

    def generate_fake_samples(self, input_images):
        input_handle, loss_handle, eval_handle, summary_handle = self.retrieve_handle()
        fake_images, real_images, d_loss, g_loss, l1_loss = self.sess.run([eval_handle.generator,
                                                                           eval_handle.target,
                                                                           loss_handle.d_loss,
                                                                           loss_handle.g_loss,
                                                                           loss_handle.l1_loss],
                                                                          feed_dict={
                                                                              input_handle.real_data: input_images
                                                                          })
        return fake_images, real_images, d_loss, g_loss, l1_loss

    def validate_model(self, images, epoch, step):
        fake_imgs, real_imgs, d_loss, g_loss, l1_loss = self.generate_fake_samples(images)
        print("Sample: d_loss: %.5f, g_loss: %.5f, l1_loss:%.5f" % (d_loss, g_loss, l1_loss))

        merged_fake_images = merge(scale_back(fake_imgs), [self.batch_size, 1])
        merged_real_images = merge(scale_back(real_imgs), [self.batch_size, 1])
        merged_pair = np.concatenate([merged_real_images, merged_fake_images], axis=1)

        model_id, _ = self.get_model_id_and_dir()
        model_sample_dir = os.path.join(self.sample_dir, model_id)
        if not os.path.exists(model_sample_dir):
            os.mkdir(model_sample_dir)
        sample_img_path = os.path.join(model_sample_dir, "sample_%02d_%04d.png" % (epoch, step))
        misc.imsave(sample_img_path, merged_pair)

    def export_generator(self, save_dir, model_dir, model_name="gen_model"):
        saver = tf.train.Saver()
        self.restore_model(saver, model_dir)

        gen_saver = tf.train.Saver(var_list=self.retrieve_generator_vars())
        gen_saver.save(self.sess, os.path.join(save_dir, model_name), global_step=0)

    def infer(self, source_obj, model_dir, save_dir):
        source_provider = InjectDataProvider(source_obj)
        source_iter = source_provider.get_iter(self.batch_size)

        tf.global_variables_initializer().run()

        saver = tf.train.Saver(var_list=self.retrieve_generator_vars())
        self.restore_model(saver, model_dir)

        def save_imgs(imgs, count):
            p = os.path.join(save_dir, "inferred_%04d.png" % count)
            save_concat_images(imgs, img_path=p)
            print("generated images save at %s" % p)

        count = 0
        batch_buffer = list()
        for source_imgs in source_iter:
            fake_imgs = self.generate_fake_samples(source_imgs)[0]
            merged_fake_images = merge(scale_back(fake_imgs), [self.batch_size, 1])
            batch_buffer.append(merged_fake_images)
            if len(batch_buffer) == 10:
                save_imgs(batch_buffer, count)
                batch_buffer = list()
            count += 1
        if batch_buffer:
            # last batch
            save_imgs(batch_buffer, count)

    def train(self, lr=0.0002, epoch=100, schedule=10, resume=True, freeze_encoder=False, sample_steps=1500,
              checkpoint_steps=15000):
        g_vars, d_vars = self.retrieve_trainable_vars(freeze_encoder=freeze_encoder)
        input_handle, loss_handle, _, summary_handle = self.retrieve_handle()

        if not self.sess:
            raise Exception("no session registered")

        tf.set_random_seed(1234)

        learning_rate = tf.placeholder(tf.float32, name="learning_rate")
        d_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(loss_handle.d_loss, var_list=d_vars)
        g_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(loss_handle.g_loss, var_list=g_vars)

        tf.global_variables_initializer().run()

        real_data = input_handle.real_data

        data_provider = TrainDataProvider(self.data_dir)
        total_batches = data_provider.compute_total_batch_num(self.batch_size)
        val_batch_iter = data_provider.get_val(size=self.batch_size)

        saver = tf.train.Saver(max_to_keep=3)
        summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        if resume:
            _, model_dir = self.get_model_id_and_dir()
            self.restore_model(saver, model_dir)

        current_lr = lr
        counter = 0
        start_time = time.time()

        for ei in range(epoch):
            train_batch_iter = data_provider.get_train_iter(self.batch_size)

            if (ei + 1) % schedule == 0:
                update_lr = current_lr / 2.0
                # minimum learing rate guarantee
                update_lr = max(update_lr, 0.0002)
                print("decay learning rate from %.5f to %.5f" % (current_lr, update_lr))
                current_lr = update_lr

            for bid, batch in enumerate(train_batch_iter):
                counter += 1
                batch_images = batch

                # Optimize D
                _, batch_d_loss, d_summary = self.sess.run([d_optimizer, loss_handle.d_loss, summary_handle.d_merged],
                                                           feed_dict={real_data: batch_images,
                                                                      learning_rate: current_lr})

                # Optimze G
                _, batch_g_loss = self.sess.run([g_optimizer, loss_handle.g_loss],
                                                feed_dict={
                                                    real_data: batch_images,
                                                    learning_rate: current_lr
                                                })
                # magic move to optimize G again
                # according to https://github.com/carpedm20/DCGAN-tensorflow
                # collect all the losses along the way
                _, batch_g_loss, \
                const_loss, cheat_loss, l1_loss, tv_loss, g_summary = self.sess.run([g_optimizer,
                                                                                     loss_handle.g_loss,
                                                                                     loss_handle.const_loss,
                                                                                     loss_handle.cheat_loss,
                                                                                     loss_handle.l1_loss,
                                                                                     loss_handle.tv_loss,
                                                                                     summary_handle.g_merged],
                                                                                    feed_dict={
                                                                                        real_data: batch_images,
                                                                                        learning_rate: current_lr
                                                                                    })
                passed = time.time() - start_time
                log_format = "Epoch: [%2d], [%4d/%4d] time: %4.4f, d_loss: %.5f, g_loss: %.5f, " + \
                             "const_loss: %.5f, cheat_loss: %.5f, l1_loss: %.5f, tv_loss: %.5f"
                print(log_format % (ei, bid, total_batches, passed, batch_d_loss, batch_g_loss,
                                    const_loss, cheat_loss, l1_loss, tv_loss))
                summary_writer.add_summary(d_summary, counter)
                summary_writer.add_summary(g_summary, counter)

                if counter % sample_steps == 0:
                    # sample the current model states with val data
                    self.validate_model(val_batch_iter, ei, counter)

                if counter % checkpoint_steps == 0:
                    print("Checkpoint: save checkpoint step %d" % counter)
                    self.checkpoint(saver, counter)
        # save the last checkpoint
        print("Checkpoint: last checkpoint step %d" % counter)
        self.checkpoint(saver, counter)

    def test(self, source_provider, model_dir, save_dir):
        source_len = len(source_provider.data.examples)
        source_len = min(10, source_len)

        source_iter = source_provider.get_iter(source_len)

        tf.global_variables_initializer().run()

        saver = tf.train.Saver(var_list=self.retrieve_generator_vars())
        self.restore_model(saver, model_dir)

        def save_imgs(imgs, count, threshold):
            p = os.path.join(save_dir, "inferred_%04d_%.2f.png" % (count, threshold))
            save_concat_images(imgs, img_path=p)
            print("generated images saved at %s" % p)

        def save_img(img, mse_diff, nrmse_diff, ssim_diff, psnr_diff):
            p = os.path.join(save_dir,
                             "cgan_patch%.4f-%.4f-%.4f-%.4f.png" % (ssim_diff, mse_diff, nrmse_diff, psnr_diff))
            save_image(img, img_path=p)
            print("generated ssim: %.4f images saved at %s" % (ssim_diff, p))

        counter = 0
        threshold = 0.1
        batch_buffer = list()

        for source_imgs in source_iter:
            fake_imgs, real_imgs, d_loss, g_loss, l1_loss = self.generate_fake_samples(source_imgs)
            img_shape = fake_imgs.shape

            fake_imgs_reshape = np.reshape(np.array(fake_imgs),
                                           [img_shape[0], img_shape[1] * img_shape[2] * img_shape[3]])
            real_imgs_reshape = np.reshape(np.array(real_imgs),
                                           [img_shape[0], img_shape[1] * img_shape[2] * img_shape[3]])

            fake_imgs_reshape_saved = fake_imgs_reshape
            real_imgs_reshape_saved = real_imgs_reshape

            # threshold -- fixed
            for bt in range(fake_imgs_reshape.shape[0]):
                for it in range(fake_imgs_reshape.shape[1]):
                    if fake_imgs_reshape[bt][it] >= threshold:
                        fake_imgs_reshape[bt][it] = 1.0
                    else:
                        fake_imgs_reshape[bt][it] = -1.0

            # otsu threshold
            # radius = 15
            # selem = disk(radius)
            #
            # local_otsu = rank.otsu(fake_imgs_reshape, selem)
            # fake_imgs_reshape >= local_otsu

            # mse, nrmse, ssim and psnr
            for bt in range(fake_imgs_reshape.shape[0]):
                mse_diff = compare_mse(real_imgs_reshape[bt], fake_imgs_reshape[bt])
                nrmse_diff = compare_nrmse(real_imgs_reshape[bt], fake_imgs_reshape[bt], norm_type="Euclidean")
                ssim_diff = compare_ssim(real_imgs_reshape[bt], fake_imgs_reshape[bt])
                psnr_diff = compare_psnr(real_imgs_reshape[bt], fake_imgs_reshape[bt])
                print("mse diff:{} | nrmse diff:{} | ssim:{} | psnr:{}".format(mse_diff, nrmse_diff,
                                                                               ssim_diff, psnr_diff))
                # save the images with ssim > 0.8 and ssim < 0.5
                if ssim_diff > 0.8 or ssim_diff < 0.5:
                    fk_reshape = np.reshape(fake_imgs_reshape_saved[bt], (1, fake_imgs.shape[1], fake_imgs.shape[2],
                                                                          fake_imgs.shape[3]))
                    rl_reshape = np.reshape(real_imgs_reshape_saved[bt], (1, real_imgs.shape[1], real_imgs.shape[2],
                                                                          real_imgs.shape[3]))
                    fk_reshape = merge(scale_back(fk_reshape), [1, 1])
                    rl_reshape = merge(scale_back(rl_reshape), [1, 1])
                    pair = np.concatenate([rl_reshape, fk_reshape], axis=1)
                    save_img(pair, mse_diff, nrmse_diff, ssim_diff, psnr_diff)

            fake_imgs_reshape = np.reshape(fake_imgs_reshape, fake_imgs.shape)
            real_imgs_reshape = np.reshape(real_imgs_reshape, real_imgs.shape)
            merged_fake_images = merge(scale_back(fake_imgs_reshape), [source_len, 1])
            merged_real_images = merge(scale_back(real_imgs_reshape), [source_len, 1])
            merged_pair = np.concatenate([merged_real_images, merged_fake_images], axis=1)

            batch_buffer.append(merged_pair)
            counter += 1
        if batch_buffer:
            # last batch
            save_imgs(batch_buffer, counter, threshold)












